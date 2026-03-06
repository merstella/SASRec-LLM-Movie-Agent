import torch
import numpy as np
import pickle
import random
import os
from tqdm import tqdm
from model import SASRec

args = {
    'maxlen': 50,
    'hidden_units': 128,
    'num_blocks': 2,
    'num_heads': 4,
    'dropout_rate': 0.2,
    'lr': 0.001,
    'batch_size': 128,
    'num_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(user_indices, train_data, num_items, maxlen):
    seqs, pos, neg = [], [], []
    for u in user_indices:
        items = train_data[u]
        if len(items) < 2:
            continue

        seq = np.zeros([maxlen], dtype=np.int32)
        p_seq = np.zeros([maxlen], dtype=np.int32)
        n_seq = np.zeros([maxlen], dtype=np.int32)
        item_set = set(items)

        idx = maxlen - 1
        for i in range(len(items) - 2, -1, -1):
            seq[idx] = items[i]
            p_seq[idx] = items[i + 1]

            neg_item = 0
            for _ in range(32):
                candidate = random.randint(1, num_items)
                if candidate not in item_set:
                    neg_item = candidate
                    break

            if neg_item == 0:
                if len(item_set) >= num_items:
                    p_seq[idx] = 0
                    n_seq[idx] = 0
                else:
                    start = random.randint(1, num_items)
                    candidate = start
                    while candidate in item_set:
                        candidate += 1
                        if candidate > num_items:
                            candidate = 1
                    n_seq[idx] = candidate
            else:
                n_seq[idx] = neg_item

            idx -= 1
            if idx == -1:
                break

        seqs.append(seq)
        pos.append(p_seq)
        neg.append(n_seq)

    return (
        np.asarray(seqs, dtype=np.int64),
        np.asarray(pos, dtype=np.int64),
        np.asarray(neg, dtype=np.int64),
    )

def train():
    set_seed(42)

    with open('data/sasrec_data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    train_data = dataset['train']
    num_items = dataset['num_items']

    model = SASRec(num_items, args).to(args['device'])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98))

    model.train()
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args['num_epochs']):
        user_indices = list(train_data.keys())
        random.shuffle(user_indices)

        epoch_loss = 0.0
        num_steps = 0
        pbar = tqdm(range(0, len(user_indices), args['batch_size']), desc=f"Epoch {epoch+1}")

        for i in pbar:
            batch_users = user_indices[i : i + args['batch_size']]
            seqs, pos, neg = get_batch(batch_users, train_data, num_items, args['maxlen'])
            if seqs.shape[0] == 0:
                continue

            optimizer.zero_grad()
            pos_logits, neg_logits = model(seqs, pos, neg)

            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            indices = torch.where(torch.from_numpy(pos).to(args['device']) != 0)
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_steps += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / max(num_steps, 1)
        print(f"Epoch {epoch+1} done, average loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'checkpoints/sasrec.pt')
    print("Saved checkpoint to checkpoints/sasrec.pt")

if __name__ == "__main__":
    train()
