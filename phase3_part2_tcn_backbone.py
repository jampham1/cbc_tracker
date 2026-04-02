"""
Phase 3 — Part 2: TCN Backbone
================================
Stacks multiple TemporalBlocks with exponentially increasing dilation
to form the full sequence encoder.

Receptive field grows as:
    RF = 1 + 2 * (kernel_size - 1) * (2^n_blocks - 1)

For kernel_size=3, n_blocks=4:
    RF = 1 + 2 * 2 * (16 - 1) = 61 timesteps
    This fully covers our window_size=15 with headroom to spare.

Design decisions explained inline.
"""

import torch
import torch.nn as nn
import math

# Import TemporalBlock from Part 1
# (adjust path if you've placed files in src/)
from phase3_part1_temporal_block import TemporalBlock


class TCNBackbone(nn.Module):
    """
    Full TCN sequence encoder.

    Stacks N TemporalBlocks where block i uses dilation = 2^i.
    Each block operates on (batch, channels, seq_len) tensors.

    Args:
        in_channels   : input feature dimension  (8: CBC features + log_time_delta)
        hidden_channels: number of channels in each hidden block
        n_blocks      : number of TemporalBlocks to stack
        kernel_size   : convolutional kernel width (3 is standard for TCN)
        dropout       : dropout probability in each block
    """

    def __init__(
        self,
        in_channels:     int   = 8,
        hidden_channels: int   = 64,
        n_blocks:        int   = 4,
        kernel_size:     int   = 3,
        dropout:         float = 0.2,
    ):
        super().__init__()

        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.n_blocks        = n_blocks

        # ── Build the stack of TemporalBlocks ─────────────────────────────────
        # Block 0 : in_channels  → hidden_channels, dilation=1
        # Block 1 : hidden_channels → hidden_channels, dilation=2
        # Block 2 : hidden_channels → hidden_channels, dilation=4
        # Block 3 : hidden_channels → hidden_channels, dilation=8
        # ...
        #
        # Only the first block changes the channel count (from in_channels to
        # hidden_channels). All subsequent blocks keep it constant, so their
        # residual projection is just nn.Identity() — no extra parameters.

        layers = []
        for i in range(n_blocks):
            in_ch    = in_channels if i == 0 else hidden_channels
            dilation = 2 ** i          # 1, 2, 4, 8, ...

            layers.append(
                TemporalBlock(
                    in_channels  = in_ch,
                    out_channels = hidden_channels,
                    kernel_size  = kernel_size,
                    dilation     = dilation,
                    dropout      = dropout,
                )
            )

        # nn.Sequential lets us call self.blocks(x) and it passes x through
        # each block in order automatically
        self.blocks = nn.Sequential(*layers)

        # ── Compute and store receptive field for reference ───────────────────
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** n_blocks - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, in_channels, seq_len)
                  Note: Conv1d expects channels BEFORE seq_len.
                  Your DataLoader gives (batch, seq_len, features) —
                  you'll need to permute before calling this.
                  We handle that in the full model (Part 4).
        Returns:
            out : (batch, hidden_channels, seq_len)
                  Every output timestep has attended to all past inputs
                  within the receptive field — causally.
        """
        return self.blocks(x)

    def describe(self):
        """Print a human-readable summary of the backbone."""
        print("── TCN Backbone ────────────────────────────────────────")
        print(f"  Input channels   : {self.in_channels}")
        print(f"  Hidden channels  : {self.hidden_channels}")
        print(f"  Blocks           : {self.n_blocks}")
        print(f"  Dilations        : {[2**i for i in range(self.n_blocks)]}")
        print(f"  Receptive field  : {self.receptive_field} timesteps")

        total = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters : {total:,}")
        print()

        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            print(f"  Block {i}  dilation={2**i:<4}  params={block_params:,}")
        print("────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Receptive field calculator — useful when tuning architecture
# ─────────────────────────────────────────────────────────────────────────────
def compute_receptive_field(kernel_size: int, n_blocks: int) -> int:
    """
    Returns the receptive field size for a TCN with the given config.

    Rule of thumb: receptive field should be >= your window_size.
    More is fine — the model simply won't use the extra capacity.
    Less means some positions at the start of a window can't attend
    far enough back, which limits what the early layers can learn.
    """
    return 1 + 2 * (kernel_size - 1) * (2 ** n_blocks - 1)


def suggest_n_blocks(window_size: int, kernel_size: int = 3) -> int:
    """
    Returns the minimum number of blocks needed for the receptive field
    to cover the full window.
    """
    n = 1
    while compute_receptive_field(kernel_size, n) < window_size:
        n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── 1. Print receptive field table ───────────────────────────────────────
    print("Receptive field by number of blocks (kernel_size=3):")
    print(f"  {'Blocks':<8} {'Dilations':<20} {'Receptive field'}")
    for n in range(1, 7):
        dilations = [2**i for i in range(n)]
        rf = compute_receptive_field(kernel_size=3, n_blocks=n)
        print(f"  {n:<8} {str(dilations):<20} {rf} timesteps")

    print()
    min_blocks = suggest_n_blocks(window_size=15, kernel_size=3)
    print(f"Minimum blocks to cover window_size=15: {min_blocks}")
    print(f"We use n_blocks=4  →  RF={compute_receptive_field(3,4)} (comfortable headroom)")
    print()

    # ── 2. Forward pass shape check ───────────────────────────────────────────
    backbone = TCNBackbone(
        in_channels     = 8,    # CBC features + log_time_delta
        hidden_channels = 64,
        n_blocks        = 4,
        kernel_size     = 3,
        dropout         = 0.2,
    )
    backbone.describe()

    batch_size = 32
    seq_len    = 15
    x   = torch.randn(batch_size, 8, seq_len)   # (batch, channels, seq_len)
    out = backbone(x)

    print(f"\nForward pass:")
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Expected: ({batch_size}, 64, {seq_len})")

    assert out.shape == (batch_size, 64, seq_len), "Shape mismatch!"
    print("TCNBackbone shape check passed.")

    # ── 3. Check causality — output at t should not depend on t+1 ─────────────
    # Method: run two inputs that are identical except at the last timestep.
    # The outputs at ALL timesteps except the last should be identical.
    # If they're not, the convolution is leaking future information.
    # Must be in eval() mode — Dropout is stochastic in train mode, which causes
    # outputs to differ between two forward passes even with identical inputs.
    # eval() disables Dropout, making the network fully deterministic.
    print("\nCausality check...")
    backbone.eval()
    x1 = torch.randn(1, 8, seq_len)
    x2 = x1.clone()
    x2[:, :, -1] = x2[:, :, -1] + 999.0   # dramatically change last timestep

    with torch.no_grad():
        out1 = backbone(x1)
        out2 = backbone(x2)

    diff = (out1 - out2).abs()
    max_diff_early = diff[:, :, :-1].max().item()   # all but last timestep
    max_diff_last  = diff[:, :, -1].max().item()    # only the last timestep

    print(f"  Max diff at t<T (should be ~0): {max_diff_early:.6f}")
    print(f"  Max diff at t=T (should be >0): {max_diff_last:.4f}")
    assert max_diff_early < 1e-4, "Causality violated — model sees the future!"
    print("Causality check passed.")