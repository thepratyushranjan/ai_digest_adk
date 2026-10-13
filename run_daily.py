import asyncio
import logging

from dotenv import load_dotenv

load_dotenv("ai_digest/.env")

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from ai_digest.agent import root_agent  # noqa: E402

APP, USER = "ai_digest", "me"


async def main() -> None:
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=APP, user_id=USER)
    runner = Runner(app_name=APP, agent=root_agent, session_service=sessions)
    start = types.Content(role="user", parts=[types.Part(text="Run today's digest.")])

    async for event in runner.run_async(user_id=USER, session_id=session.id, new_message=start):
        if event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts).strip()
            if text:
                print(f"[{event.author}] {text[:200]}")

    final = await sessions.get_session(app_name=APP, user_id=USER, session_id=session.id)
    print("\nDone:", final.state.get("delivery_status"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "google_genai", "google_adk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(main())
