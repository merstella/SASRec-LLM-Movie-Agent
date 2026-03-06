from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommendation.agents import agent_utils

uid = 1

print("--- 1. Test History ---")
history = agent_utils.get_user_history(uid, n=3)
print(history)

print("\n--- 2. Test Recommendation + Rerank ---")
candidates = agent_utils.recommend_next_candidates(uid, k=10)
final_list = agent_utils.rerank_with_query(candidates, "space adventure", top_n=3)
details = agent_utils.get_item_details(final_list)
print(details)
