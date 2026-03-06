import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from movie_recommendation.models.model import SASRec

DATA_DIR = Path("data")
CHECKPOINT_PATH = Path("checkpoints/sasrec.pt")
MODEL_ARGS = {
    "maxlen": 50,
    "hidden_units": 128,
    "num_blocks": 2,
    "num_heads": 4,
    "dropout_rate": 0.2,
    "device": "cpu",
}

text_model = SentenceTransformer("all-MiniLM-L6-v2")

with open(DATA_DIR / "sasrec_data.pkl", "rb") as f:
    dataset = pickle.load(f)
    train_data, val_data, num_items = dataset["train"], dataset["val"], dataset["num_items"]

with open(DATA_DIR / "id2item.json", "r") as f:
    id2item = json.load(f)

items_meta = pd.read_json(DATA_DIR / "item_meta.json")
item_emb_matrix = np.load(DATA_DIR / "item_emb.npy")

sasrec_model = SASRec(num_items, MODEL_ARGS)
sasrec_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
sasrec_model.eval()


def _to_original_item_ids(item_ids: list[int]) -> list[int]:
    return [int(id2item[str(item_id)]) for item_id in item_ids if str(item_id) in id2item]


def get_user_history(user_id: int, n: int = 15) -> list[dict[str, Any]]:
    u_idx = int(user_id)
    if u_idx not in train_data:
        return []

    history_ids = train_data[u_idx] + val_data.get(u_idx, [])
    original_ids = _to_original_item_ids(history_ids[-n:])
    history_info = items_meta[items_meta["itemId"].isin(original_ids)]
    return history_info[["title", "genres"]].to_dict(orient="records")


def recommend_next_candidates(user_id: int, k: int = 50) -> list[int]:
    u_idx = int(user_id)
    if u_idx not in train_data:
        return []

    history = train_data[u_idx] + val_data.get(u_idx, [])
    if not history:
        return []

    seq = np.zeros([MODEL_ARGS["maxlen"]], dtype=np.int32)
    idx = MODEL_ARGS["maxlen"] - 1
    for item_id in reversed(history):
        seq[idx] = item_id
        idx -= 1
        if idx == -1:
            break

    with torch.no_grad():
        input_seq = torch.from_numpy(np.array([seq])).long()
        log_feats = sasrec_model.forward_eval(input_seq)
        final_feat = log_feats[:, -1, :].numpy()

        scores = np.matmul(final_feat, sasrec_model.item_emb.weight.detach().numpy().T).squeeze()
        scores[np.asarray(history, dtype=np.int64)] = -1e9
        scores[0] = -1e9
        top_k_indices = np.argsort(scores)[-k:][::-1]

    return top_k_indices.tolist()


def rerank_with_query(candidate_ids: list[int], query: str, top_n: int = 10) -> list[int]:
    if not candidate_ids:
        return []
    if not query:
        return candidate_ids[:top_n]

    query_vector = text_model.encode([query])
    candidate_vectors = item_emb_matrix[np.asarray(candidate_ids, dtype=np.int64)]
    similarities = cosine_similarity(query_vector, candidate_vectors).flatten()
    sorted_idx = np.argsort(similarities)[::-1]
    reranked_ids = [candidate_ids[i] for i in sorted_idx]
    return reranked_ids[:top_n]


def get_item_details(item_ids: list[int]) -> list[dict[str, Any]]:
    if not item_ids:
        return []

    original_ids = _to_original_item_ids(item_ids)
    details = items_meta[items_meta["itemId"].isin(original_ids)]
    return details[["itemId", "title", "genres"]].to_dict(orient="records")
