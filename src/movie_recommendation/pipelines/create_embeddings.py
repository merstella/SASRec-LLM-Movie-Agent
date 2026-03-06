import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
ITEM_DATA_PATH = DATA_DIR / "items.parquet"
ITEM2ID_PATH = DATA_DIR / "item2id.json"


def create_item_embeddings():
    items_df = pd.read_parquet(ITEM_DATA_PATH)
    with open(ITEM2ID_PATH, "r") as f:
        item2id = json.load(f)

    items_df["combined_text"] = items_df["title"] + " [" + items_df["genres"] + "]"
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = items_df["combined_text"].tolist()
    print(f"Creating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True)

    num_items = len(item2id)
    embedding_dim = embeddings.shape[1]
    final_embeddings = np.zeros((num_items + 1, embedding_dim), dtype=np.float32)

    item_index = {int(item_id): idx for idx, item_id in enumerate(items_df["itemId"].tolist())}
    for old_id, new_id in item2id.items():
        idx_in_df = item_index[int(old_id)]
        final_embeddings[int(new_id)] = embeddings[idx_in_df]

    np.save(DATA_DIR / "item_emb.npy", final_embeddings)
    items_df[["itemId", "title", "genres"]].to_json(DATA_DIR / "item_meta.json", orient="records")
    print(f"Completed. Embedding matrix shape: {final_embeddings.shape}")


if __name__ == "__main__":
    create_item_embeddings()
