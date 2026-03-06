import json
import pickle
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "interactions.parquet"
MIN_INTERACTIONS = 3


def preprocess_sasrec():
    df = pd.read_parquet(RAW_DATA_PATH)
    df = df.reset_index(names="_row_id")
    df = df.sort_values(["userId", "timestamp", "itemId", "_row_id"], kind="mergesort")
    df = df.drop(columns=["_row_id"])

    user_ids = df["userId"].unique()
    item_ids = df["itemId"].unique()

    user2id = {int(old): i + 1 for i, old in enumerate(user_ids)}
    item2id = {int(old): i + 1 for i, old in enumerate(item_ids)}
    id2item = {i + 1: int(old) for i, old in enumerate(item_ids)}

    with open(DATA_DIR / "user2id.json", "w") as f:
        json.dump(user2id, f)
    with open(DATA_DIR / "item2id.json", "w") as f:
        json.dump(item2id, f)
    with open(DATA_DIR / "id2item.json", "w") as f:
        json.dump(id2item, f)

    df["u_idx"] = df["userId"].map(user2id)
    df["i_idx"] = df["itemId"].map(item2id)
    user_group = df.groupby("u_idx")["i_idx"].apply(list)

    train_set = {}
    val_set = {}
    test_set = {}
    num_items = len(item_ids)
    removed_users = 0

    for u_idx, items in user_group.items():
        if len(items) < MIN_INTERACTIONS:
            removed_users += 1
            continue

        train_set[u_idx] = items[:-2]
        val_set[u_idx] = [items[-2]]
        test_set[u_idx] = [items[-1]]

    dataset = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "num_items": num_items,
    }

    with open(DATA_DIR / "sasrec_data.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(
        f"Done! Eligible users: {len(train_set)}, "
        f"removed users (<{MIN_INTERACTIONS} interactions): {removed_users}, "
        f"Total items: {num_items}"
    )


if __name__ == "__main__":
    preprocess_sasrec()
