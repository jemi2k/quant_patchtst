"""
smoke_test.py
=============
Self-contained smoke test for the Dual-Head PatchTST + PCGrad training step.

Verifies the mathematical graph end-to-end on synthetic data (no real market data):
  1. Forward pass emits RAW LOGITS of shape [Batch, Channels, 2].
  2. pcgrad_train_step() executes both backward passes + PCGrad surgery without
     shape / graph errors, and returns finite losses.
  3. model.attns is populated correctly: a list of None when
     output_attention=False, real attention matrices when True.

Run from the repo root:  python smoke_test.py

SHAPE CONTRACT (documented so the future dataloaders don't trip on it):
  - Input  x    : [Batch, seq_len, Channels]   (RevIN's [B, T, C] convention)
  - Labels y    : [Batch, Channels, 2]         (long, short) per asset
  - Weights u_t : [Batch, Channels]            per (batch, asset) sample
  The `Channels` axis sits on a DIFFERENT position for input vs output. That is
  expected -- Model.forward permutes internally -- so dataloaders must produce x
  and y in these exact shapes.
"""

import math

import torch

from configs import Config
from models.PatchTST import Model
from train_step import pcgrad_train_step


def make_dummy(batch, seq_len, channels):
    """Build synthetic input / labels / weights of the correct shapes."""
    x = torch.randn(batch, seq_len, channels)            # [B, T, C]
    y = (torch.rand(batch, channels, 2) > 0.5).float()   # [B, C, 2] binary 0/1
    u_t = torch.rand(batch, channels)                    # [B, C] uniqueness in (0, 1)
    return x, y, u_t


def main():
    torch.manual_seed(0)

    batch, channels = 4, 7
    cfg = Config()  # defaults: seq_len=96, patch_len=16, stride=8, padding=0
    patch_num = (cfg.seq_len + cfg.padding - cfg.patch_len) // cfg.stride + 1  # 11

    x, y, u_t = make_dummy(batch, cfg.seq_len, channels)

    # --- 1. Forward pass: logit shape & finiteness ---------------------------
    model = Model(cfg)
    model.train()
    logits = model(x)
    assert logits.shape == (batch, channels, 2), f"logits {tuple(logits.shape)} != ({batch},{channels},2)"
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    print(f"[PASS] forward -> logits shape {tuple(logits.shape)}, all finite")

    # --- 2. Training step: dual BCE + PCGrad backward -------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = pcgrad_train_step(model, optimizer, x, y, u_t)
    assert set(losses) == {"long", "short", "total"}, f"unexpected loss keys {list(losses)}"
    assert all(math.isfinite(v) for v in losses.values()), f"non-finite loss {losses}"
    print(f"[PASS] pcgrad_train_step -> long={losses['long']:.5f}, short={losses['short']:.5f}, total={losses['total']:.5f}")

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"param {name} went NaN/Inf after PCGrad"
    print("[PASS] all parameters finite after PCGrad step")

    # --- 3. Attention stash ----------------------------------------------------
    # (a) default config: output_attention=False -> list of None
    assert len(model.attns) == cfg.e_layers, f"attns length {len(model.attns)} != {cfg.e_layers}"
    assert all(a is None for a in model.attns), "expected None attns when output_attention=False"
    print(f"[PASS] output_attention=False -> attns = {len(model.attns)}x None")

    # (b) output_attention=True -> real attention matrices
    cfg_attn = Config(output_attention=True)
    model_attn = Model(cfg_attn)
    model_attn.train()
    _ = model_attn(x)
    attn_shape = (batch * channels, cfg_attn.n_heads, patch_num, patch_num)
    assert len(model_attn.attns) == cfg_attn.e_layers, "attns length mismatch"
    for a in model_attn.attns:
        assert a is not None and tuple(a.shape) == attn_shape, f"attn {None if a is None else a.shape} != {attn_shape}"
    print(f"[PASS] output_attention=True -> attns shape {attn_shape}")

    print("\nAll smoke tests passed -- Phase 1 is wired correctly.")


if __name__ == "__main__":
    main()
