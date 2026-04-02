"""
Phase 3 — Part 4: Full Model + Anomaly Head
=============================================
Wires TCNBackbone + StaticFeatureBranch into a single nn.Module
and adds a pointwise conv head that outputs a per-timestep
anomaly probability.

Data flow:
    x_seq    (batch, T, 8)   ← from DataLoader
        └─ permute → (batch, 8, T)
        └─ TCNBackbone → (batch, 64, T)
        └─ + static_embedding (batch, 64, 1) [broadcast]
        └─ fused (batch, 64, T)
        └─ AnomalyHead Conv1d(64→1, k=1) → (batch, 1, T)
        └─ squeeze → (batch, T)
        └─ sigmoid → (batch, T)  ← anomaly probability per timestep

    x_static (batch, 4)
        └─ StaticFeatureBranch → (batch, 64, 1)
"""

import torch
import torch.nn as nn

from phase3_part2_tcn_backbone import TCNBackbone
from phase3_part3_static_branch import StaticFeatureBranch


class CBCAnomalyTCN(nn.Module):
    """
    Full longitudinal CBC anomaly detector.

    Args:
        n_seq_features    : number of sequential input features
                            (7 CBC + 1 log_time_delta = 8)
        n_static_features : number of static patient features (4)
        hidden_channels   : TCN and static branch channel width
        n_blocks          : number of TemporalBlocks in the backbone
        kernel_size       : TCN conv kernel size
        dropout           : dropout rate used in backbone and static branch
    """

    def __init__(
        self,
        n_seq_features:    int   = 8,
        n_static_features: int   = 4,
        hidden_channels:   int   = 64,
        n_blocks:          int   = 4,
        kernel_size:       int   = 3,
        dropout:           float = 0.2,
    ):
        super().__init__()

        # ── Part 2: TCN backbone ──────────────────────────────────────────────
        self.backbone = TCNBackbone(
            in_channels     = n_seq_features,
            hidden_channels = hidden_channels,
            n_blocks        = n_blocks,
            kernel_size     = kernel_size,
            dropout         = dropout,
        )

        # ── Part 3: Static feature branch ────────────────────────────────────
        self.static_branch = StaticFeatureBranch(
            in_features     = n_static_features,
            hidden_channels = hidden_channels,
            dropout         = dropout / 2,
        )

        # ── Anomaly head ──────────────────────────────────────────────────────
        # kernel_size=1 is a pointwise conv — no temporal look-back.
        # It simply projects each timestep's 64-channel representation
        # down to a single scalar. All temporal reasoning was already
        # done by the backbone; this head just reads off the result.
        self.anomaly_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1),
        )

        # Sigmoid converts raw logit → probability in [0, 1]
        # NOTE: we keep sigmoid here for inference.
        # During training with BCEWithLogitsLoss we'll skip it
        # (that loss applies sigmoid internally for numerical stability).
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x_seq:    torch.Tensor,   # (batch, T, n_seq_features)
        x_static: torch.Tensor,   # (batch, n_static_features)
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_seq         : (batch, T, n_seq_features)
                            Note the T-before-features layout from DataLoader.
                            We permute to (batch, features, T) for Conv1d.
            x_static      : (batch, n_static_features)
            return_logits : if True, return raw scores before sigmoid.
                            Set True during training with BCEWithLogitsLoss.

        Returns:
            anomaly_prob  : (batch, T)
                            Per-timestep anomaly probability (or logit).
        """

        # ── Step 1: permute sequence for Conv1d ───────────────────────────────
        # DataLoader gives us (batch, T, features) — time-first layout.
        # Conv1d expects (batch, features, T) — channel-first layout.
        x = x_seq.permute(0, 2, 1)          # (batch, 8, T)

        # ── Step 2: TCN backbone ──────────────────────────────────────────────
        x = self.backbone(x)                 # (batch, 64, T)

        # ── Step 3: static feature fusion ────────────────────────────────────
        static_emb = self.static_branch(x_static)   # (batch, 64, 1)
        x = x + static_emb                           # (batch, 64, T) via broadcast

        # ── Step 4: anomaly head ──────────────────────────────────────────────
        x = self.anomaly_head(x)             # (batch, 1, T)
        x = x.squeeze(1)                     # (batch, T)

        # ── Step 5: activation ───────────────────────────────────────────────
        if return_logits:
            return x                         # raw scores for BCEWithLogitsLoss
        return self.sigmoid(x)               # probabilities for inference

    def count_parameters(self) -> dict:
        """Break down parameter count by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        total    = count(self)
        backbone = count(self.backbone)
        static   = count(self.static_branch)
        head     = count(self.anomaly_head)

        return {
            'backbone':      backbone,
            'static_branch': static,
            'anomaly_head':  head,
            'total':         total,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BATCH   = 32
    T       = 15    # window size
    N_SEQ   = 8     # CBC features + log_time_delta
    N_STAT  = 4     # static patient features

    model = CBCAnomalyTCN(
        n_seq_features    = N_SEQ,
        n_static_features = N_STAT,
        hidden_channels   = 64,
        n_blocks          = 4,
        kernel_size       = 3,
        dropout           = 0.2,
    )

    # ── 1. Shape check ────────────────────────────────────────────────────────
    x_seq    = torch.randn(BATCH, T, N_SEQ)
    x_static = torch.randn(BATCH, N_STAT)

    probs  = model(x_seq, x_static)
    logits = model(x_seq, x_static, return_logits=True)

    print('── Shape checks ────────────────────────────────────────')
    print(f'  x_seq input    : {tuple(x_seq.shape)}')
    print(f'  x_static input : {tuple(x_static.shape)}')
    print(f'  output (probs) : {tuple(probs.shape)}')
    print(f'  output (logits): {tuple(logits.shape)}')
    print(f'  expected output: ({BATCH}, {T})')

    assert probs.shape  == (BATCH, T), "Probability output shape wrong"
    assert logits.shape == (BATCH, T), "Logit output shape wrong"
    print('  Shape check passed.\n')

    # ── 2. Probability range check ────────────────────────────────────────────
    print('── Probability range check ─────────────────────────────')
    print(f'  Min prob : {probs.min().item():.4f}  (should be >= 0.0)')
    print(f'  Max prob : {probs.max().item():.4f}  (should be <= 1.0)')
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of [0,1]"
    print('  Range check passed.\n')

    # ── 3. Parameter count breakdown ─────────────────────────────────────────
    counts = model.count_parameters()
    print('── Parameter breakdown ─────────────────────────────────')
    for name, n in counts.items():
        bar = '█' * (n // 1000)
        print(f'  {name:<16} {n:>8,}  {bar}')
    print()

    # ── 4. Causality check ───────────────────────────────────────────────────
    print('── Causality check ─────────────────────────────────────')
    model.eval()
    x1 = torch.randn(1, T, N_SEQ)
    x2 = x1.clone()
    x2[:, -1, :] += 999.0    # change only the last timestep

    with torch.no_grad():
        out1 = model(x1, x_static[:1])
        out2 = model(x2, x_static[:1])

    diff           = (out1 - out2).abs()
    max_diff_early = diff[:, :-1].max().item()
    max_diff_last  = diff[:,  -1].max().item()

    print(f'  Max diff at t < T (should be ~0): {max_diff_early:.8f}')
    print(f'  Max diff at t = T (should be > 0): {max_diff_last:.6f}')
    assert max_diff_early < 1e-5, "Causality violated!"
    assert max_diff_last  > 0.0,  "Model insensitive to last timestep"
    print('  Causality check passed.\n')

    # ── 5. Gradient flow check ────────────────────────────────────────────────
    # Verify gradients reach all the way back to the input.
    # If any component has a broken connection, its gradients will be None.
    print('── Gradient flow check ─────────────────────────────────')
    model.train()
    x_seq_g    = torch.randn(BATCH, T, N_SEQ,  requires_grad=True)
    x_static_g = torch.randn(BATCH, N_STAT,    requires_grad=True)

    out  = model(x_seq_g, x_static_g, return_logits=True)
    loss = out.mean()
    loss.backward()

    assert x_seq_g.grad    is not None, "No gradient to x_seq!"
    assert x_static_g.grad is not None, "No gradient to x_static!"
    print(f'  x_seq grad norm    : {x_seq_g.grad.norm().item():.6f}')
    print(f'  x_static grad norm : {x_static_g.grad.norm().item():.6f}')
    print('  Gradient flow check passed.')
    print('───────────────────────────────────────────────────────')
    print('\nAll checks passed. Model is ready for Phase 4 training.')
