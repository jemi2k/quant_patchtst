import torch
import torch.nn as nn

from layers.RevIN import RevIN
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import EncoderLayer, Encoder

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        the Prediction Head: Flattens the processed patches and projects them to the forecast horizon.
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        # Expected input x: [Batch, Channels, d_model, Num_Patches]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in

        # 1. Reversible Normalization Layer
        self.revin_layer = RevIN(num_features=self.n_vars, affine=configs.affine)

        # 2. Patching & Embedding Layer
        # calculate how many patches we will have based on sequence length, patch length, and stride.
        patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 1)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, configs.patch_len, configs.stride, configs.padding, configs.dropout
        )

        # 3. transformer Backbone (The Encoder Stack)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), 
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 4. Prediction Head
        self.head = FlattenHead(
            configs.enc_in, 
            nf=configs.d_model * patch_num, 
            target_window=configs.pred_len, 
            head_dropout=configs.dropout
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        The flow of data through the architecture. 
        (Note: x_mark_enc, x_dec, etc., are standard TSLib inputs, but PatchTST ignores them).
        """
        # --- PHASE 1: Normalization ---
        # Input shape: [Batch, seq_len, Channels]
        x = self.revin_layer(x, mode='norm')

        # --- PHASE 2: Patching & Embedding ---
        # PatchEmbedding expects [Batch, Channels, seq_len], so we permute.
        x = x.permute(0, 2, 1) 
        
        # Embed! Output shape: [Batch * Channels, patch_num, d_model]
        x, n_vars = self.patch_embedding(x)

        # --- PHASE 3: Transformer Encoder ---
        # The data passes through the attention blocks
        x, attns = self.encoder(x)

        # --- PHASE 4: Preparation for the Head ---
        # Current shape: [Batch * Channels, patch_num, d_model]
        # We must pull Batch and Channels back apart to predict each stream cleanly.
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])  # [Batch, Channels, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)                            # [Batch, Channels, d_model, patch_num]

        # --- PHASE 5: Prediction & Denormalization ---
        # Forecast the future! Output shape: [Batch, Channels, pred_len]
        dec_out = self.head(x) 
        
        # RevIN expects [Batch, pred_len, Channels], so we permute back
        dec_out = dec_out.permute(0, 2, 1)
        
        # Denormalize to restore absolute price values
        dec_out = self.revin_layer(dec_out, mode='denorm')

        return dec_out