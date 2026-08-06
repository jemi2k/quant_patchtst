import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        a single Transformer Encoder block.
        """
        super(EncoderLayer, self).__init__()
        # d_ff is the hidden dimension of the MLP (usually 4x d_model)
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        
        # --- THE FEED-FORWARD NETWORK (MLP) ---
        # i just used 1D Convolutions with kernel_size=1. mathematically, 
        # this is identical to an nn.Linear layer applied independently to each patch.
        # IF we TEST KAN LATER, THIS IS WHAT we REPLACE.
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Normalization and Regularization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 1. Multi-Head Attention from the SelfAttention_Family.py module
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        
        # 2. First Residual Connection & Layer Normalization
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        
        # 3. The Feed-Forward Network (MLP)
        # We must transpose because nn.Conv1d expects shape [Batch, Channels, Length]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 4. Second Residual Connection & Layer Normalization
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        the stack that holds multiple EncoderLayers together (usually 3 layers deep).
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        
        # Pass the data sequentially through Layer 1, then Layer 2, then Layer 3...
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        # final Layer Normalization over the whole stack
        if self.norm is not None:
            x = self.norm(x)

        return x, attns