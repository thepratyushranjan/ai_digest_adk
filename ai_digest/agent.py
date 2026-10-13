from google.adk.agents import SequentialAgent

from .collection import collection_stage
from .stages import delivery_agent, filter_agent, top10_agent, write_and_review

root_agent = SequentialAgent(
    name="daily_ai_digest",
    description="Collect -> filter -> top 10 -> write & review -> email.",
    sub_agents=[
        collection_stage,   # 1. Data Collection
        filter_agent,       # 2. Data Filtering
        top10_agent,        # 3. Top 10 Selection
        write_and_review,   # 4. Content Aggregation + 5. Review (loop)
        delivery_agent,     # Final output: save + email
    ],
)
