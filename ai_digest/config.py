import os
from pathlib import Path
from zoneinfo import ZoneInfo

from google.adk.models import Gemini
from google.genai import types

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
HISTORY_FILE = DATA_DIR / "history.json"
HISTORY_DAYS = 7
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "36"))
USER_AGENT = "PersonalAIDigest/1.0 (personal news reader)"

# Models: a fast/cheap one for bulk filtering, a stronger one for judgment.
FAST_MODEL = os.getenv("FAST_MODEL", "gemini-3.5-flash-lite")
SMART_MODEL = os.getenv("SMART_MODEL", "gemini-3.8-flash")
SEARCH_MODEL = os.getenv("SEARCH_MODEL", SMART_MODEL)


def llm(name: str) -> Gemini:
    """Model with retry + exponential backoff on 429 (quota) and 5xx errors."""
    return Gemini(model=name, retry_options=types.HttpRetryOptions(
        attempts=6, initial_delay=5, max_delay=60, exp_base=2, jitter=1))

# (display name, feed URL, max items from this feed)
FEEDS = [
    # ("OpenAI News", "https://openai.com/news/rss.xml", 10),
    # ("Google DeepMind", "https://deepmind.google/blog/rss.xml", 10),
    ("Google AI Blog", "https://blog.google/technology/ai/rss/", 10),
    # ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", 10),
    # ("Microsoft Research", "https://www.microsoft.com/en-us/research/feed/", 8),
    # ("NVIDIA Blog", "https://blogs.nvidia.com/feed/", 8),
    # ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", 15),
    # ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", 15),
    # ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed", 10),
    # ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/technology-lab", 10),
    # ("arXiv cs.AI", "https://rss.arxiv.org/rss/cs.AI", 10),
    # ("arXiv cs.CL", "https://rss.arxiv.org/rss/cs.CL", 8),
    # ("arXiv cs.LG", "https://rss.arxiv.org/rss/cs.LG", 8),
    # ("Hacker News", "https://hnrss.org/frontpage?points=150", 15),
]

# Each inner list becomes one web search agent; they run in parallel.
SEARCH_TOPIC_GROUPS = [
    ["New AI model releases and major updates (Anthropic, OpenAI, Google, Meta, xAI, Mistral, DeepSeek, Qwen)",
     "Major AI product launches and features from big tech companies"],
    ["AI chips, GPUs, and AI infrastructure / data center news",
     "Important open-source AI releases (models, frameworks, agent tools)"],
    ["AI policy, regulation, and legal developments (US, EU, India)",
     "Significant AI startup funding rounds or acquisitions",
     "Notable AI research breakthroughs covered by the press"],
]

# What you care about. The filtering and top 10 agents use this.
INTEREST_PROFILE = os.getenv("INTEREST_PROFILE", """\
I follow AI and the broader tech industry. Most interested in: new models and
capabilities, AI agents and developer tools, open-source AI, AI hardware and
chips, major product launches, and policy that affects builders.
Less interested in: celebrity/opinion takes, minor funding rounds, crypto,
generic "AI will change everything" think pieces.""")
