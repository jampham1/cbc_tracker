"""
Phase 3 — Part 3: Static Feature Branch
=========================================
A small MLP that encodes patient-level static covariates
(age, sex, cancer type, chemo protocol) into a fixed-size
embedding vector that gets fused with the TCN's sequence output.

Why a separate branch?
  Static features don't change over time — feeding them as an
  extra channel in the sequence would repeat the same value at
  every timestep, wasting capacity and potentially confusing the
  convolutional layers which are designed to detect *changes*.
  A separate MLP learns a compact representation of the patient
  context, then adds it as a bias term across the full sequence.

Fusion strategy — additive injection:
  After the MLP produces a vector of shape (batch, hidden_channels),
  we unsqueeze it to (batch, hidden_channels, 1) and add it to the
  TCN output of shape (batch, hidden_channels, T).
  PyTorch broadcasting handles the T dimension automatically —
  the same patient context vector shifts every timestep equally,
  like a learned per-patient bias on top of the sequence encoding.
"""

import torch
import torch.nn as nn


class StaticFeatureBranch(nn.Module):
    """
    MLP encoder for static patient covariates.

    Args:
        in_features     : number of static input features (4 in our case:
                          age_norm, sex_enc, cancer_type_enc, chemo_protocol_enc)
        hidden_channels : output size — must match TCN backbone hidden_channels
                          so the two can be added together
        dropout         : dropout probability between MLP layers
    """

    def __init__(
        self,
        in_features:     int   = 4,
        hidden_channels: int   = 64,
        dropout:         float = 0.1,
    ):
        super().__init__()

        # Two-layer MLP:
        #   Layer 1 : in_features → hidden_channels * 2  (expand)
        #   Layer 2 : hidden_channels * 2 → hidden_channels  (project to match TCN)
        #
        # The expansion in layer 1 gives the MLP room to learn non-linear
        # interactions between static features (e.g. age + cancer type together)
        # before compressing back to the target size.
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_static : (batch, in_features)
        Returns:
            embedding : (batch, hidden_channels, 1)
                        The trailing 1 allows broadcasting across the
                        time dimension when added to TCN output.
        """
        out = self.mlp(x_static)        # (batch, hidden_channels)
        return out.unsqueeze(-1)         # (batch, hidden_channels, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    batch_size      = 32
    n_static        = 4
    hidden_channels = 64
    seq_len         = 15

    branch = StaticFeatureBranch(
        in_features     = n_static,
        hidden_channels = hidden_channels,
    )

    x_static = torch.randn(batch_size, n_static)
    embedding = branch(x_static)

    print(f'Static input shape : {tuple(x_static.shape)}')
    print(f'Embedding shape    : {tuple(embedding.shape)}')
    print(f'Expected           : ({batch_size}, {hidden_channels}, 1)')
    assert embedding.shape == (batch_size, hidden_channels, 1)
    print('Shape check passed.\n')

    # ── Fusion simulation ─────────────────────────────────────────────────────
    # Simulate what happens when we add this to the TCN output.
    # The (batch, hidden_channels, 1) embedding broadcasts across seq_len.
    tcn_output  = torch.randn(batch_size, hidden_channels, seq_len)
    fused       = tcn_output + embedding   # broadcasting: 1 expands to seq_len

    print(f'TCN output shape   : {tuple(tcn_output.shape)}')
    print(f'After fusion shape : {tuple(fused.shape)}')
    assert fused.shape == (batch_size, hidden_channels, seq_len)
    print('Fusion broadcast check passed.\n')

    # ── Verify broadcast is correct ───────────────────────────────────────────
    # eval() disables Dropout so the MLP is deterministic.
    # We then check that (fused - tcn_output) is identical at every timestep
    # meaning the same embedding bias was added across the full sequence.
    # atol=1e-6 accounts for normal float32 precision variation.
    branch.eval()
    with torch.no_grad():
        embedding2    = branch(x_static)
        fused2        = tcn_output + embedding2

    diff = fused2 - tcn_output   # (batch, hidden_channels, seq_len)
    first_timestep = diff[:, :, 0]
    for t in range(1, seq_len):
        assert torch.allclose(diff[:, :, t], first_timestep, atol=1e-6), \
            f"Timestep {t} has a different shift — broadcast is wrong"
    print('Broadcast consistency check passed.')
    print('The same patient context vector was added at every timestep.')

    n_params = sum(p.numel() for p in branch.parameters())
    print(f'\nParameters in static branch: {n_params:,}')