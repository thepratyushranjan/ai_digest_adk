from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.adk.tools import exit_loop
from google.genai import types
from pydantic import BaseModel, Field

from . import config
from .utils import fetch_article, load_history, log


def _as_dict(value) -> dict:
    """Structured outputs may be stored as a dict or a JSON string."""
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}


def _today() -> str:
    return datetime.now(config.IST).strftime("%A, %d %B %Y")


# Filtering
class KeptItem(BaseModel):
    id: str
    relevance: int = Field(description="1-10, how relevant and important for this reader")
    reason: str = Field(description="A few words on why it was kept")


class FilterResult(BaseModel):
    keep: list[KeptItem]


def filter_instruction(ctx: ReadonlyContext) -> str:
    items = [{k: i[k] for k in ("id", "title", "source", "published", "snippet")}
             for i in ctx.state.get("collected_items", [])]
    return f"""You are the Filtering Agent. Today is {_today()}.
Reader profile:
{config.INTEREST_PROFILE}

From the items below, KEEP only items that are:
- relevant to the reader profile,
- real news or research (not opinion, listicles, ads, or SEO filler),
- from a credible source, and not near-duplicates of another kept item (keep the best source).
Aim to keep 30-60 items. Score relevance 1-10.

Items (JSON):
{json.dumps(items, ensure_ascii=False)}"""


filter_agent = LlmAgent(
    name="filter_agent",
    description="Filters collected items by relevance and quality.",
    model=config.llm(config.FAST_MODEL),
    instruction=filter_instruction,
    output_schema=FilterResult,
    output_key="filtered",
    include_contents="none",
)


# Top 10
class Story(BaseModel):
    headline: str
    why_important: str
    item_ids: list[str] = Field(description="ids of all items covering this story")


class Top10Result(BaseModel):
    stories: list[Story]


def top10_instruction(ctx: ReadonlyContext) -> str:
    by_id = {i["id"]: i for i in ctx.state.get("collected_items", [])}
    kept = _as_dict(ctx.state.get("filtered")).get("keep", [])
    candidates = [{**{k: by_id[k_["id"]][k] for k in ("id", "title", "source", "snippet")},
                   "relevance": k_["relevance"]}
                  for k_ in kept if k_["id"] in by_id]
    recent = "\n".join(f"- {h['date']}: {h['headline']}" for h in load_history()) or "(none)"
    return f"""You are the Top 10 Selection Agent. Today is {_today()}.
Reader profile:
{config.INTEREST_PROFILE}

1. Group candidate items that cover the same event into one story.
2. Pick the 10 most important stories for this reader.
3. Skip stories already sent recently (list below) unless there is a major new development.
Prefer diversity: don't fill the list with one company or one topic.

Recently sent headlines:
{recent}

Candidates (JSON):
{json.dumps(candidates, ensure_ascii=False)}"""


top10_agent = LlmAgent(
    name="top10_agent",
    description="Groups items into stories and selects the top 10.",
    model=config.llm(config.SMART_MODEL),
    instruction=top10_instruction,
    output_schema=Top10Result,
    output_key="top10",
    include_contents="none",
)


# Write and review loop
def _stories_with_sources(ctx: ReadonlyContext) -> str:
    by_id = {i["id"]: i for i in ctx.state.get("collected_items", [])}
    stories = _as_dict(ctx.state.get("top10")).get("stories", [])
    out = []
    for n, s in enumerate(stories, 1):
        srcs = [by_id[i] for i in s.get("item_ids", []) if i in by_id]
        out.append({"n": n, "headline": s["headline"], "why_important": s["why_important"],
                    "sources": [{k: x[k] for k in ("title", "url", "source", "snippet")} for x in srcs]})
    return json.dumps(out, ensure_ascii=False, indent=1)


def aggregation_instruction(ctx: ReadonlyContext) -> str:
    notes = ctx.state.get("review_notes", "")
    revise = (f"\nA reviewer checked your previous draft and requested these fixes:\n{notes}\n"
              f"Previous draft:\n{ctx.state.get('draft', '')}\nApply the fixes and output the full revised digest."
              if notes and ctx.state.get("draft") else "")
    return f"""You are the Content Aggregation Agent. Today is {_today()}.
Write ONE concise, accurate daily digest of the 10 stories below.

Use the fetch_article tool to read the most useful 1-2 sources per story before writing.
Only state facts found in the sources. No speculation or hype.

Format (Markdown):
# AI & Tech Digest — {_today()}
Then for each story:
## <n>. <headline>
2-4 sentence summary.
**Why it matters:** one sentence.
Sources: markdown links.
{revise}
Stories and sources:
{_stories_with_sources(ctx)}"""


aggregation_agent = LlmAgent(
    name="aggregation_agent",
    description="Reads sources and writes the digest draft.",
    model=config.llm(config.SMART_MODEL),
    instruction=aggregation_instruction,
    tools=[fetch_article],
    output_key="draft",
    include_contents="none",
)


def review_instruction(ctx: ReadonlyContext) -> str:
    return f"""You are the Review Agent. Check the draft digest below for accuracy and conciseness.

Check: names, numbers, dates, and claims match the sources (use fetch_article to verify
anything doubtful); no story is duplicated; each summary is 2-4 sentences; every source link
is from the provided source list.

If the draft is accurate and concise: call the exit_loop tool, then reply APPROVED.
Otherwise: reply with a short numbered list of specific fixes. Do NOT rewrite the draft yourself.

Draft:
{ctx.state.get('draft', '')}

Stories and allowed sources:
{_stories_with_sources(ctx)}"""


review_agent = LlmAgent(
    name="review_agent",
    description="Fact-checks the draft; approves it or sends back fixes.",
    model=config.llm(config.SMART_MODEL),
    instruction=review_instruction,
    tools=[fetch_article, exit_loop],
    output_key="review_notes",
    include_contents="none",
)

write_and_review = LoopAgent(
    name="write_and_review",
    description="Draft, review, revise. Stops when approved or after 2 rounds.",
    sub_agents=[aggregation_agent, review_agent],
    max_iterations=2,
)


# Delivery
class DeliveryAgent(BaseAgent):
    """Saves the digest, updates history.json, and emails it (plain code, no LLM)."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        digest = ctx.session.state.get("draft", "")
        today = datetime.now(config.IST)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (config.DATA_DIR / f"digest_{today:%Y-%m-%d}.md").write_text(digest, encoding="utf-8")

        # Remember today's headlines for the next 7 days.
        stories = _as_dict(ctx.session.state.get("top10")).get("stories", [])
        keep_after = (today - timedelta(days=config.HISTORY_DAYS)).strftime("%Y-%m-%d")
        history = [h for h in load_history() if h["date"] >= keep_after]
        history += [{"date": f"{today:%Y-%m-%d}", "headline": s["headline"]} for s in stories]
        config.HISTORY_FILE.write_text(json.dumps(history, indent=1, ensure_ascii=False), encoding="utf-8")

        status = self._send_email(digest, today) if digest else "No digest to send."
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    content=types.Content(role="model", parts=[types.Part(text=status)]),
                    actions=EventActions(state_delta={"delivery_status": status}))

    @staticmethod
    def _send_email(digest: str, today: datetime) -> str:
        sender, password = os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_APP_PASSWORD")
        to = os.getenv("EMAIL_TO", sender)
        if not (sender and password):
            return "Email settings missing; digest saved to file only."
        import markdown  # converts the Markdown digest to HTML

        msg = MIMEText(markdown.markdown(digest), "html", "utf-8")
        msg["Subject"] = f"AI & Tech Digest — {today:%d %b %Y}"
        msg["From"], msg["To"] = sender, to
        try:
            with smtplib.SMTP_SSL(os.getenv("SMTP_HOST", "smtp.gmail.com"), 465, timeout=30) as smtp:
                smtp.login(sender, password)
                smtp.send_message(msg)
            return f"Digest emailed to {to}."
        except (smtplib.SMTPException, OSError) as exc:
            log.error("Email failed: %s", exc)
            return f"Email failed ({exc}); digest saved to file."


delivery_agent = DeliveryAgent(name="delivery_agent", description="Saves and emails the digest.")
