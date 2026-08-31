import torch
import torch.nn as nn

from layers.RevIN import RevIN
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import EncoderLayer, Encoder


class DualHead(nn.Module):
    """Two independent binary classifiers sharing one flattened encoder output.

    - Bull head -> P(long / upper barrier touched first)
    - Bear head -> P(short / lower barrier touched first)

    Each head is a single Linear over the flattened [d_model, patch_num]
    representation, which preserves the exact temporal geometry of every patch
    (we deliberately avoid global-average-pooling for this reason). Heads emit
    RAW LOGITS -- the sigmoid is folded into BCEWithLogitsLoss in train_step.py
    for numerical stability (log-sum-exp).
    """

    def __init__(self, nf, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)   # [..., d_model, patch_num] -> [..., d_model * patch_num]
        self.dropout = nn.Dropout(dropout)
        self.bull_head = nn.Linear(nf, 1)
        self.bear_head = nn.Linear(nf, 1)

    def forward(self, x):
        # x: [Batch, Channels, d_model, patch_num]
        x = self.flatten(x)                       # [Batch, Channels, d_model * patch_num]
        x = self.dropout(x)
        bull = self.bull_head(x)                  # raw logit for P(long)  [Batch, Channels, 1]
        bear = self.bear_head(x)                  # raw logit for P(short) [Batch, Channels, 1]
        return torch.cat([bull, bear], dim=-1)    # [Batch, Channels, 2] raw logits


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.n_vars = configs.enc_in

        # 1. Reversible Normalization (input-only: denorm is dead code for classification)
        self.revin_layer = RevIN(num_features=self.n_vars, affine=configs.affine)

        # 2. Patching & Embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride, configs.padding, configs.dropout
        )

        # 3. Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 4. Dual classification head.
        #    The flattened input dim must equal the ACTUAL patch count produced by
        #    PatchEmbedding, which pads by `padding` and THEN unfolds:
        #        patch_num = (seq_len + padding - patch_len) // stride + 1
        #    (the first formula omitted `+ padding`, so it under-counted whenever
        #    padding > 0 and caused a Linear shape mismatch.)
        patch_num = (configs.seq_len + configs.padding - configs.patch_len) // configs.stride + 1
        self.head = DualHead(nf=configs.d_model * patch_num, dropout=configs.dropout)

    def forward(self, x):
        # x: [Batch, seq_len, Channels]
        x = self.revin_layer(x, mode='norm')            # [Batch, seq_len, Channels]
        x = x.permute(0, 2, 1)                          # [Batch, Channels, seq_len]
        x, n_vars = self.patch_embedding(x)             # [Batch * Channels, patch_num, d_model]
        x, attns = self.encoder(x)                      # [Batch * Channels, patch_num, d_model]

        # Unfold Channel Independence back into per-asset streams.
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])  # [Batch, Channels, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)                            # [Batch, Channels, d_model, patch_num]

        logits = self.head(x)                            # [Batch, Channels, 2] raw logits

        # Stash attention weights for downstream explainability. Holds the per-layer
        # attention matrices when output_attention=True, else a list of None.
        self.attns = attns

        return logits
