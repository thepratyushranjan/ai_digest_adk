# Personal AI & Tech Digest (Google ADK + Gemini)

Five agents run every day at 12:00 PM IST and email you the top 10 AI/tech stories.

```
daily_ai_digest (SequentialAgent)
├─ collection_stage (SequentialAgent)        1. Data Collection
│   ├─ rss_collector            plain code
│   ├─ web_search_team (ParallelAgent)       Gemini + google_search, 3 in parallel
│   └─ collection_merger        verify links, merge, dedupe
├─ filter_agent                               2. Data Filtering   (gemini-3.5-flash-lite)
├─ top10_agent                                3. Top 10 Selection (gemini-3.8-flash)
├─ write_and_review (LoopAgent, max 2)
│   ├─ aggregation_agent                      4. Content Aggregation (reads full articles)
│   └─ review_agent                           5. Review: approve (exit_loop) or send fixes
└─ delivery_agent               plain code: save digest, update history, email
```

## Setup
```bash
pip install -r requirements.txt
cp ai_digest/.env.example ai_digest/.env    # add your Gemini key and email settings
```

## Run
```bash
python run_daily.py   # full run, headless
adk web               # dev UI: pick "ai_digest", type "run", inspect every agent step
```

## Schedule (free, no server)
Push this folder to a **private** GitHub repo, add the four secrets
(GOOGLE_API_KEY, EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_TO) under
Settings → Secrets and variables → Actions. The workflow in
`.github/workflows/daily_digest.yml` runs at 12:00 PM IST daily.

## Customize
Everything lives in `ai_digest/config.py`: feeds, search topics, models, and your
interest profile (the filtering and top 10 agents rank against it).
