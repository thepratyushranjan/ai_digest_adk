from __future__ import annotations
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from urllib.parse import urlsplit

from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.adk.models import LlmResponse
from google.adk.tools import google_search
from google.genai import types

from . import config
from .utils import collect_rss, log, make_item, normalize_url, parse_json_array, probe_url


def _cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=config.LOOKBACK_HOURS)


def _message(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])

class RssCollectorAgent(BaseAgent):
    """Fetches the fixed RSS feed list. Deterministic: no decisions, so no LLM."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        items = await asyncio.to_thread(collect_rss, _cutoff())
        headlines = "\n".join(f"- {i['title']}" for i in items[:80]) or "(none)"
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=_message(f"Collected {len(items)} items from RSS feeds."),
            actions=EventActions(state_delta={"rss_items": items, "rss_headlines": headlines}),
        )

SEARCH_INSTRUCTION = """You are a web search agent in a personal daily AI and tech news pipeline.
Your only job is to FIND recent, important news on your assigned topics. Do not rank or summarize at length.

Today is {today} (IST). Only include items published in the last {hours} hours.

Your topics:
{topics}

Headlines already collected from RSS (do not repeat these):
{known}

Rules:
- Prefer primary sources (official company blogs, papers, GitHub releases), then reputable outlets.
- Give the direct article URL. Never guess or construct a URL.
- Skip opinion pieces, listicles, SEO filler, and older news.

Respond with ONLY a JSON array (no prose, no code fences). Each element:
{{"title": str, "url": str, "source": str, "published": "YYYY-MM-DD" or null,
"snippet": "one sentence in your own words"}}
Aim for 5-12 items."""


def make_search_agent(idx: int, topics: list[str]) -> LlmAgent:
    def instruction(ctx: ReadonlyContext) -> str:
        return SEARCH_INSTRUCTION.format(
            today=datetime.now(config.IST).strftime("%A, %d %B %Y"),
            hours=config.LOOKBACK_HOURS,
            topics="\n".join(f"- {t}" for t in topics),
            known=ctx.state.get("rss_headlines", "(none)"),
        )

    def capture_grounding(callback_context: CallbackContext, llm_response: LlmResponse):
        """Save Gemini's real search source links, used later to verify URLs."""
        gm = llm_response.grounding_metadata
        if gm and gm.grounding_chunks:
            uris = [c.web.uri for c in gm.grounding_chunks if c.web and c.web.uri]
            key = f"grounding_{idx}"
            callback_context.state[key] = list(callback_context.state.get(key, [])) + uris
        return None  # keep the model's response unchanged

    return LlmAgent(
        name=f"web_search_{idx}",
        description=f"Searches the web for: {'; '.join(topics)}",
        model=config.llm(config.SEARCH_MODEL),
        instruction=instruction,
        tools=[google_search],
        include_contents="none",
        output_key=f"web_raw_{idx}",
        after_model_callback=capture_grounding,
    )


# Merge + verify
class CollectionMergerAgent(BaseAgent):
    """Verifies search-agent links, then merges and dedupes all collected items."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        n_groups = len(config.SEARCH_TOPIC_GROUPS)
        proposed, sources = [], []
        for i in range(1, n_groups + 1):
            group_items = parse_json_array(state.get(f"web_raw_{i}", ""))
            if not group_items:
                log.warning("Search group %d returned no usable items.", i)
            proposed.extend(group_items)
            sources.extend(state.get(f"grounding_{i}", []))

        web_items = await asyncio.to_thread(self._verify, proposed, sources)
        rss_items = state.get("rss_items", [])

        merged: dict[str, dict] = {}
        for item in rss_items + web_items:  # RSS version wins on duplicates
            merged.setdefault(item["id"], item)
        items = sorted(merged.values(), key=lambda i: i["published"] or "", reverse=True)

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = config.DATA_DIR / f"collected_{datetime.now(config.IST):%Y-%m-%d}.json"
        path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

        n_web = sum(1 for i in items if i["origin"] == "web_search")
        summary = (f"Collected {len(items)} unique items "
                   f"({len(items) - n_web} RSS, {n_web} new from verified web search).")
        log.info(summary)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=_message(summary),
            actions=EventActions(state_delta={"collected_items": items}),
        )

    @staticmethod
    def _verify(proposed: list[dict], sources: list[str]) -> list[dict]:
        cutoff_date = _cutoff().astimezone(config.IST).date()
        with ThreadPoolExecutor(max_workers=8) as pool:
            resolved = [r for r in pool.map(probe_url, set(sources)) if r]
        real_urls = {normalize_url(u): t for u, t in resolved}

        def verify(d: dict) -> dict | None:
            url, title = d.get("url"), d.get("title")
            if not url or not title:
                return None
            if normalize_url(url) not in real_urls:
                probed = probe_url(url)
                if not probed:
                    return None  # invented or dead link
                url = probed[0]
            published = None
            if d.get("published"):
                try:
                    published = datetime.strptime(d["published"], "%Y-%m-%d").replace(tzinfo=config.IST)
                    if published.date() < cutoff_date:
                        return None
                except ValueError:
                    pass
            return make_item(title, url, d.get("source") or urlsplit(url).netloc,
                             published, d.get("snippet", ""), "web_search")

        with ThreadPoolExecutor(max_workers=8) as pool:
            items = [i for i in pool.map(verify, proposed) if i]

        if not items and real_urls:  # model JSON unusable: fall back to raw sources
            items = [make_item(t or urlsplit(u).netloc, u, urlsplit(u).netloc, None, "", "web_search")
                     for u, t in real_urls.items()]
        return items


# Assemble stage
collection_stage = SequentialAgent(
    name="collection_stage",
    description="Collects recent AI/tech news from RSS feeds and web search.",
    sub_agents=[
        RssCollectorAgent(name="rss_collector", description="Fetches RSS feeds."),
        ParallelAgent(
            name="web_search_team",
            description="Parallel Gemini web searches, one per topic group.",
            sub_agents=[make_search_agent(i, g)
                        for i, g in enumerate(config.SEARCH_TOPIC_GROUPS, start=1)],
        ),
        CollectionMergerAgent(name="collection_merger", description="Verifies and merges items."),
    ],
)
