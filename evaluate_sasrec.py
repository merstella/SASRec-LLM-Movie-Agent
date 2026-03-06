import torch
import pickle
import numpy as np
from model import SASRec
from tqdm import tqdm

args = {
    'maxlen': 50,
    'hidden_units': 128,
    'num_blocks': 2,
    'num_heads': 4,
    'dropout_rate': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def evaluate_model(top_k=10, batch_size=512):
    print("--- Loading dataset and checkpoint ---")
    with open('data/sasrec_data.pkl', 'rb') as f:
        dataset = pickle.load(f)

    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']
    num_items = dataset['num_items']

    model = SASRec(num_items, args).to(args['device'])
    model.load_state_dict(torch.load('checkpoints/sasrec.pt', map_location=args['device']))
    model.eval()

    users = list(test_data.keys())
    seqs = np.zeros((len(users), args['maxlen']), dtype=np.int64)
    histories = []
    ground_truths = np.zeros(len(users), dtype=np.int64)

    for row, u in enumerate(users):
        history = train_data[u] + val_data[u]
        histories.append(history)
        ground_truths[row] = test_data[u][0]

        idx = args['maxlen'] - 1
        for item in reversed(history):
            seqs[row, idx] = item
            idx -= 1
            if idx == -1:
                break

    recalls = []
    ndcgs = []

    with torch.no_grad():
        item_embs = model.item_emb.weight.transpose(0, 1)

        for start in tqdm(range(0, len(users), batch_size), desc="Evaluating SASRec"):
            end = min(len(users), start + batch_size)
            batch_seq = torch.from_numpy(seqs[start:end]).long().to(args['device'])

            log_feats = model.forward_eval(batch_seq)
            final_feat = log_feats[:, -1, :]
            logits = torch.matmul(final_feat, item_embs)

            for row, history in enumerate(histories[start:end]):
                logits[row, history] = -1e9
            logits[:, 0] = -1e9

            top_indices = torch.topk(logits, top_k, dim=1).indices.cpu().numpy()

            for row, top_items in enumerate(top_indices):
                ground_truth = int(ground_truths[start + row])
                top_list = top_items.tolist()
                if ground_truth in top_list:
                    recalls.append(1.0)
                    rank = top_list.index(ground_truth)
                    ndcgs.append(1.0 / np.log2(rank + 2))
                else:
                    recalls.append(0.0)
                    ndcgs.append(0.0)

    print("\n" + "="*30)
    print("SASRec Evaluation")
    print("-" * 30)
    print(f"Recall@10: {np.mean(recalls):.4f}")
    print(f"NDCG@10:   {np.mean(ndcgs):.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()
