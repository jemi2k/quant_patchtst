import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # compute the positional encodings once in log space (standard sine/cosine)
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch * Channels, Num_Patches, Patch_Length]
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        """
        tokenizes time-series data into patches and projects them into the Transformer's hidden dimension
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # 1. padding Layer: duplicates the last few data points if the sequence 
        # doesn't divide cleanly into the patch/stride dimensions
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # 2. linear Projection: Maps the raw patch window to the d_model space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # 3. positional Encoding
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        args:
            x: Input tensor of shape [Batch, Channels, Time_Steps]
               (Note: RevIN normalized the data, and the model wrapper permuted it to this shape)
        """
        n_vars = x.shape[1]
        
        # Phase 1: Pad the sequence length
        x = self.padding_patch_layer(x)
        
        # Phase 2: Unfold the continuous time series into discrete patches
        # Transforms shape to: [Batch, Channels, Num_Patches, Patch_Length]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Phase 3: The "Channel Fold" (the secret to Channel Independence)
        # we squash the Batch and Channels together 
        # new shape: [Batch * Channels, Num_Patches, Patch_Length]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # Phase 4: Linear Projection & Positional Context
        # Project Patch_Length -> d_model and add sine/cosine wave context
        # Final shape: [Batch * Channels, Num_Patches, d_model]
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x), n_vars