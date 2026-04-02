"""
Phase 3 — Part 1: The Temporal Block
======================================
The fundamental building block of the TCN.

One TemporalBlock contains:
  - Two dilated causal Conv1d layers (with the same dilation)
  - Weight normalisation on each conv (stabilises training)
  - ReLU activation + Dropout after each conv
  - A residual (skip) connection from input to output

Why two convolutions per block?
  Two convolutions doubles the non-linearity and gives the block
  more expressive power without changing the receptive field size.
  It mirrors the pattern used in the original TCN paper
  (Bai et al. 2018).

Why a residual connection?
  Without it, stacking many layers causes vanishing gradients.
  The residual provides a gradient highway so early layers still
  receive a meaningful training signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class TemporalBlock(nn.Module):
    """
    One dilated causal convolutional block with a residual connection.

    Args:
        in_channels  : number of feature channels coming in
        out_channels : number of feature channels going out
        kernel_size  : width of the convolutional kernel (typically 3)
        dilation     : spacing between kernel elements (doubles each block)
        dropout      : dropout probability applied after each conv
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int,
        dropout:      float = 0.2,
    ):
        super().__init__()

        # ── How much left-padding do we need? ────────────────────────────────
        # For a causal conv with kernel_size=3 and dilation=2,
        # the kernel spans [t-4, t-2, t] — 4 positions back.
        # General formula: (kernel_size - 1) * dilation
        # We pad exactly this many zeros on the LEFT before each conv,
        # and zero on the right — so the conv can never see future timesteps.
        self.padding = (kernel_size - 1) * dilation

        # ── First dilated causal conv ─────────────────────────────────────────
        # padding=0 here — we apply left-only padding manually in forward()
        # using F.pad so PyTorch never pads the right side symmetrically.
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            )
        )

        # ── Second dilated causal conv ────────────────────────────────────────
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=0,
            )
        )

        # ── Activations & regularisation ─────────────────────────────────────
        self.relu1    = nn.ReLU()
        self.relu2    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # ── Residual projection ───────────────────────────────────────────────
        # If channel dimensions differ we need a 1x1 conv to match them.
        # If they're equal this is nn.Identity() — a free skip connection.
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        self.final_relu = nn.ReLU()

        # ── Weight initialisation ─────────────────────────────────────────────
        # weight_norm wraps the conv and splits its weight into two tensors:
        #   weight_v : direction  (the actual learnable parameter)
        #   weight_g : magnitude  (a scalar per output channel)
        # We must initialise weight_v, NOT .weight (which is a computed property
        # and does not retain gradients in the same way).
        nn.init.kaiming_normal_(self.conv1.parametrizations.weight.original1, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.parametrizations.weight.original1, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, in_channels, seq_len)
        Returns:
            out : (batch, out_channels, seq_len)  — same seq_len, causal
        """
        residual = x

        # ── Conv 1 ────────────────────────────────────────────────────────────
        # F.pad(tensor, (left, right)) pads the last dimension.
        # (self.padding, 0) = pad LEFT only → strictly causal.
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # ── Conv 2 ────────────────────────────────────────────────────────────
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # ── Residual ──────────────────────────────────────────────────────────
        out = self.final_relu(out + self.residual_proj(residual))

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    batch_size   = 4
    in_channels  = 8
    seq_len      = 15
    out_channels = 32

    block = TemporalBlock(
        in_channels  = in_channels,
        out_channels = out_channels,
        kernel_size  = 3,
        dilation     = 1,
        dropout      = 0.2,
    )

    # ── Shape check ───────────────────────────────────────────────────────────
    x   = torch.randn(batch_size, in_channels, seq_len)
    out = block(x)

    print(f'Input  shape : {tuple(x.shape)}')
    print(f'Output shape : {tuple(out.shape)}')
    assert out.shape == (batch_size, out_channels, seq_len), \
        "Shape mismatch — causal padding logic has a bug"
    print('Shape check passed.\n')

    # ── Causality check ───────────────────────────────────────────────────────
    # IMPORTANT: must call block.eval() before this test.
    # Dropout randomly zeros activations during training — stochastic between
    # two forward passes — so outputs will differ even with identical inputs.
    # eval() disables Dropout (and BatchNorm if present), making the network
    # fully deterministic and the diff test meaningful.
    block.eval()

    x1 = torch.randn(1, in_channels, seq_len)
    x2 = x1.clone()
    x2[:, :, -1] += 999.0   # change only the very last timestep dramatically

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    diff = (out1 - out2).abs()
    max_diff_early = diff[:, :, :-1].max().item()  # timesteps 0..T-2
    max_diff_last  = diff[:, :,  -1].max().item()  # timestep T-1 only

    print(f'Max diff at t < T (should be ~0.0) : {max_diff_early:.8f}')
    print(f'Max diff at t = T (should be > 0.0): {max_diff_last:.4f}')

    assert max_diff_early < 1e-5, "Causality violated — model sees the future!"
    assert max_diff_last  > 0.0,  "Last timestep shows no sensitivity — something is wrong"
    print('Causality check passed.')

    n_params = sum(p.numel() for p in block.parameters())
    print(f'\nParameters in this block: {n_params:,}')