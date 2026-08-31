"""
configs.py
==========
Single source of truth for the PatchTST *architecture* hyperparameters.

This dataclass defines the model shape only. Data, labeling, and validation
hyperparameters (DIB thresholds, fractional-diff order, triple-barrier widths,
CPCV N/k, etc.) belong to later phases and will live in their own modules —
we deliberately do NOT mix "model geometry" with "data/validation" concerns.

Usage:
    from configs import Config
    cfg = Config()                  # smoke-test defaults
    cfg.seq_len = 128               # override per experiment
    model = Model(cfg)
"""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Architecture configuration for the Channel-Independent PatchTST.

    All sequence lengths below are measured in *Information Time* — DIB bars,
    not wall-clock seconds. A window of `seq_len` bars is the model's lookback.
    """

    # --- Input / window geometry -------------------------------------------------
    seq_len: int = 96        # lookback window: number of DIB bars fed to the encoder
    patch_len: int = 16      # length of each patch (bars per patch token)
    stride: int = 8          # step between patch starts (overlap = patch_len - stride)
    padding: int = 0         # right-side replication padding applied before patching

    # --- Embedding & encoder -----------------------------------------------------
    d_model: int = 128       # hidden dimension of each patch token
    n_heads: int = 8         # attention heads (d_model must divide evenly by n_heads)
    d_ff: int = 256          # feed-forward hidden dimension (the MLP KAN will replace later)
    e_layers: int = 3        # number of stacked encoder layers
    dropout: float = 0.1     # dropout applied to attention, FFN, and patch embedding
    activation: str = "gelu" # FFN activation: "relu" or "gelu"
    affine: bool = False     # RevIN learnable scale/shift (True adds 2 * enc_in params)
    output_attention: bool = False  # return encoder attention weights (for explainability)

    # --- Channel / output ---------------------------------------------------------
    enc_in: int = 7          # number of channels = the 7 L1/Infrastructure futures assets
    n_classes: int = 2       # dual-head output dim: [P(long), P(short)] per asset

    def to_dict(self) -> dict:
        """serialisable snapshot, used for logging & experiment reproducibility."""
        return asdict(self)
