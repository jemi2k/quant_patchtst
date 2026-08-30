"""
train_step.py
=============
Custom training step for the Dual-Head PatchTST.

Implements:
  1. Dual Binary Cross-Entropy via `BCEWithLogitsLoss` with per-sample
     uniqueness weights u_t (reduction='none'), so overlapping / highly
     correlated samples contribute less to the gradient.
  2. PCGrad (Gradient Surgery) on the shared Transformer trunk: when the Bull
     and Bear gradients conflict, each is projected onto the other's normal
     plane before summing, preventing the two heads from cannibalising the
     shared representation.

The model emits raw logits [Batch, Channels, 2]; the sigmoid is folded into
BCEWithLogitsLoss here (log-sum-exp trick), keeping the forward pass numerically
stable and free of vanishing-gradient NaNs.
"""

import torch
import torch.nn as nn


def _weighted_bce(logits, targets, weights, criterion):
    """scale-invariant, per-sample weighted BCE

    logits : [Batch, Channels] raw logits
    targets: [Batch, Channels] binary labels (0.0 / 1.0)
    weights: [Batch, Channels] per-sample uniqueness weights u_t
    criterion: nn.BCEWithLogitsLoss(reduction='none')

    returns a scalar: sum(loss * u_t) / sum(u_t). Dividing by sum(u_t) makes this
    a proper weighted mean, invariant to the absolute scale of u_t -- only the
    relative weighting across samples matters 
    """
    loss = criterion(logits, targets)              # [Batch, Channels]
    loss = loss * weights                          # downweight overlapping samples
    return loss.sum() / (weights.sum() + 1e-8)     # +eps guards against all-zero u_t


def pcgrad_train_step(model, optimizer, x, y, u_t, criterion=None):
    """run one dual-task training step with PCGrad gradient surgery.

    Args:
        model    : Dual-Head PatchTST. forward(x) -> raw logits [Batch, Channels, 2].
        optimizer: torch optimizer built over model.parameters().
        x        : [Batch, seq_len, Channels] input windows.
        y        : [Batch, Channels, 2] binary labels (long, short).
        u_t      : [Batch, Channels] per-sample uniqueness weights.
        criterion: nn.BCEWithLogitsLoss(reduction='none'); created if None.

    Returns:
        dict of detached scalar losses: {'long', 'short', 'total'}.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    # --- 1. Forward & dual losses -------------------------------------------------
    logits = model(x)                              # [Batch, Channels, 2] raw logits
    y_long  = y[..., 0]                            # [Batch, Channels]
    y_short = y[..., 1]                            # [Batch, Channels]

    loss_long  = _weighted_bce(logits[..., 0], y_long,  u_t, criterion)
    loss_short = _weighted_bce(logits[..., 1], y_short, u_t, criterion)

    # --- 2. Partition parameters: shared trunk vs the two heads -------------------
    shared = [p for n, p in model.named_parameters() if not n.startswith('head.')]
    bull   = [p for n, p in model.named_parameters() if n.startswith('head.bull_head')]
    bear   = [p for n, p in model.named_parameters() if n.startswith('head.bear_head')]

    # --- 3. Backward pass #1 (Bull task) ------------------------------------------
    # retain_graph=True keeps the shared trunk alive so the Bear loss can reuse it.
    optimizer.zero_grad()
    loss_long.backward(retain_graph=True)
    g_long    = [p.grad.clone() for p in shared]   # trunk grads induced by Bull loss
    bull_grad = [p.grad.clone() for p in bull]     # Bull head's own grads

    # --- 4. Backward pass #2 (Bear task) ------------------------------------------
    optimizer.zero_grad()
    loss_short.backward(retain_graph=True)
    g_short    = [p.grad.clone() for p in shared]  # trunk grads induced by Bear loss
    bear_grad  = [p.grad.clone() for p in bear]    # Bear head's own grads

    # --- 5. PCGrad: resolve conflicting trunk gradients ---------------------------
    trunk_grad = []
    for gl, gs in zip(g_long, g_short):
        gl_flat = gl.reshape(-1)
        gs_flat = gs.reshape(-1)
        dot = torch.dot(gl_flat, gs_flat)

        if dot < 0:
            # Conflict: project each gradient onto the other's normal plane.
            gl_proj = gl_flat - (dot / (gs_flat.norm() ** 2 + 1e-8)) * gs_flat
            gs_proj = gs_flat - (dot / (gl_flat.norm() ** 2 + 1e-8)) * gl_flat
            trunk_grad.append((gl_proj + gs_proj).reshape_as(gl))
        else:
            # No conflict: ordinary sum.
            trunk_grad.append((gl_flat + gs_flat).reshape_as(gl))

    # --- 6. Reassign gradients and step -------------------------------------------
    for p, g in zip(shared, trunk_grad):
        p.grad = g
    for p, g in zip(bull, bull_grad):
        p.grad = g
    for p, g in zip(bear, bear_grad):
        p.grad = g

    optimizer.step()

    return {
        'long':  loss_long.item(),
        'short': loss_short.item(),
        'total': loss_long.item() + loss_short.item(),
    }
