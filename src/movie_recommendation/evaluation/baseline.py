import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_data():
    with open("data/sasrec_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data["train"], data["val"], data["test"], data["num_items"]


def evaluate(recommendations, ground_truth):
    if ground_truth in recommendations:
        rank = recommendations.index(ground_truth)
        return 1.0, 1.0 / np.log2(rank + 2)
    return 0.0, 0.0


def get_user_history(train, val, u_idx):
    return train.get(u_idx, []) + val.get(u_idx, [])


def filter_seen_from_ranked(ranked_items, seen_items, top_k):
    top_items = []
    for item in ranked_items:
        if item in seen_items:
            continue
        top_items.append(item)
        if len(top_items) == top_k:
            break
    return top_items


def popularity_baseline(train, val, test, num_items, top_k=10):
    all_items = []
    for items in train.values():
        all_items.extend(items)

    ranked_items = [item for item, _ in Counter(all_items).most_common()]
    recalls, ndcgs = [], []
    for u_idx, gt_list in test.items():
        seen_items = set(get_user_history(train, val, u_idx))
        top_items = filter_seen_from_ranked(ranked_items, seen_items, top_k)
        recall, ndcg = evaluate(top_items, gt_list[0])
        recalls.append(recall)
        ndcgs.append(ndcg)

    return np.mean(recalls), np.mean(ndcgs)


def item_knn_baseline(train, val, test, num_items, top_k=10):
    num_users = max(train.keys()) + 1
    matrix = np.zeros((num_users, num_items + 1), dtype=np.float32)
    for u_idx, items in train.items():
        for item_idx in items:
            matrix[u_idx, item_idx] = 1

    item_sim = cosine_similarity(matrix.T)

    all_items = []
    for items in train.values():
        all_items.extend(items)
    pop_ranked_items = [item for item, _ in Counter(all_items).most_common()]

    recalls, ndcgs = [], []
    for u_idx, gt_list in tqdm(test.items(), desc="Running Item-KNN"):
        user_history = get_user_history(train, val, u_idx)
        if not user_history:
            top_items = filter_seen_from_ranked(pop_ranked_items, set(), top_k)
            recall, ndcg = evaluate(top_items, gt_list[0])
            recalls.append(recall)
            ndcgs.append(ndcg)
            continue

        scores = item_sim[user_history].sum(axis=0)
        scores[user_history] = -1
        scores[0] = -1

        top_items = np.argsort(scores)[-top_k:][::-1].tolist()
        recall, ndcg = evaluate(top_items, gt_list[0])
        recalls.append(recall)
        ndcgs.append(ndcg)

    return np.mean(recalls), np.mean(ndcgs)


if __name__ == "__main__":
    train, val, test, num_items = load_data()

    print("--- Popularity Baseline ---")
    pop_r, pop_n = popularity_baseline(train, val, test, num_items)

    print("--- Item-KNN Baseline ---")
    knn_r, knn_n = item_knn_baseline(train, val, test, num_items)

    results = pd.DataFrame(
        {
            "Metric": ["Recall@10", "NDCG@10"],
            "Popularity": [pop_r, pop_n],
            "Item-KNN": [knn_r, knn_n],
        }
    )
    print("\nBASELINE RESULT:")
    print(results.to_string(index=False))
