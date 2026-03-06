import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.dev = args['device']

        self.item_emb = nn.Embedding(self.item_num + 1, args['hidden_units'], padding_idx=0)
        self.pos_emb = nn.Embedding(args['maxlen'], args['hidden_units'])
        self.emb_dropout = nn.Dropout(p=args['dropout_rate'])

        self.attention_layernorms = nn.ModuleList([
            nn.LayerNorm(args['hidden_units'], eps=1e-8) for _ in range(args['num_blocks'])
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                args['hidden_units'],
                args['num_heads'],
                dropout=args['dropout_rate'],
                batch_first=True
            )
            for _ in range(args['num_blocks'])
        ])
        self.forward_layernorms = nn.ModuleList([
            nn.LayerNorm(args['hidden_units'], eps=1e-8) for _ in range(args['num_blocks'])
        ])
        self.forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args['hidden_units'], args['hidden_units']),
                nn.ReLU(),
                nn.Linear(args['hidden_units'], args['hidden_units'])
            ) for _ in range(args['num_blocks'])
        ])

        self.last_layernorm = nn.LayerNorm(args['hidden_units'], eps=1e-8)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _as_long_tensor(self, seqs):
        if torch.is_tensor(seqs):
            return seqs.long().to(self.dev)
        return torch.as_tensor(seqs, dtype=torch.long, device=self.dev)

    def forward_eval(self, log_seqs):
        # Shared sequence encoder used for both train and eval.
        log_seqs = self._as_long_tensor(log_seqs)

        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        batch_size, seq_len = log_seqs.shape
        positions = torch.arange(seq_len, device=self.dev).unsqueeze(0).expand(batch_size, seq_len)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs.eq(0)
        seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        attention_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev),
            diagonal=1
        )

        for i in range(len(self.attention_layers)):
            normalized_seqs = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                normalized_seqs,
                normalized_seqs,
                normalized_seqs,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask,
                need_weights=False
            )
            seqs = seqs + mha_outputs

            seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.forward_eval(log_seqs)  # (Batch, MaxLen, Hidden)

        pos_embs = self.item_emb(self._as_long_tensor(pos_seqs))  # (Batch, MaxLen, Hidden)
        neg_embs = self.item_emb(self._as_long_tensor(neg_seqs))  # (Batch, MaxLen, Hidden)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits
