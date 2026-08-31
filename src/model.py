"""
World model for Trackmania: recurrent dynamics with material-aware
PointNet track encoder.

Architecture:
  - State encoder: MLP(STATE_DIM → hidden)
  - Action encoder: MLP(ACTION_DIM → 32)
  - Track encoder: PointNet with material embedding → track_out
  - Recurrence: GRUCell (hidden + 32 + track_out → hidden)
  - Prediction head: MLP(hidden → TARGET_DIM)

GRUCell (not GRU) is used for explicit per-step control during
autoregressive rollout — the training loop manages the sequence,
not PyTorch.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from config import RunConfig
from features import STATE_DIM, ACTION_DIM, TARGET_DIM


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * num_layers + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrackPointEncoder(nn.Module):
    """
    Material-aware PointNet encoder for track geometry patches.

    Each point is (x, y, z, material_id).  The material ID is embedded
    via a learned lookup table, then concatenated with xyz before the
    per-point MLP.  Global feature is computed via max-pooling.

    Input:  (B, N_POINTS, 4) — [x, y, z, material_id]
    Output: (B, out_dim)
    """

    def __init__(
        self,
        num_materials: int,
        material_dim: int = 4,
        point_hidden: int = 64,
        out_dim: int = 64,
    ):
        super().__init__()
        self.material_embed = nn.Embedding(num_materials, material_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 + material_dim, point_hidden),
            nn.ReLU(),
            nn.Linear(point_hidden, point_hidden),
            nn.ReLU(),
            nn.Linear(point_hidden, out_dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ctx: (B, N_POINTS, 4) — [x, y, z, material_id]

        Returns:
            (B, out_dim) global track feature
        """
        xyz = ctx[..., :3]
        mat = ctx[..., 3].long()
        mat_emb = self.material_embed(mat)          # (B, N, material_dim)
        feat = torch.cat([xyz, mat_emb], dim=-1)    # (B, N, 3+material_dim)
        per_point = self.mlp(feat)                   # (B, N, out_dim)
        return per_point.max(dim=1).values           # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────
# World model
# ──────────────────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    GRUCell-based world model.

    The training loop calls forward() once per timestep, managing
    the hidden state explicitly.  This gives full control over
    teacher forcing, rollout length, and hidden state manipulation.

    Input per step:
        state:     (B, STATE_DIM)  — normalized car state
        action:    (B, ACTION_DIM) — normalized action
        track_ctx: (B, N_POINTS, 4) — local point cloud [xyz, material]
        hidden:    (B, H) — GRU hidden state

    Output:
        new_hidden: (B, H)
        delta_pred: (B, TARGET_DIM) — predicted normalized state delta
    """

    def __init__(
        self,
        num_materials: int = 1,
        hidden_dim: int = 256,
        track_out_dim: int = 64,
        material_dim: int = 4,
    ):
        super().__init__()

        self.state_enc = MLP(STATE_DIM, 128, hidden=128, num_layers=2)
        self.action_enc = MLP(ACTION_DIM, 32, hidden=32, num_layers=2)
        self.track_enc = TrackPointEncoder(
            num_materials=num_materials,
            material_dim=material_dim,
            out_dim=track_out_dim,
        )

        gru_input_dim = 128 + 32 + track_out_dim
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(gru_input_dim, hidden_dim)

        self.pred_head = MLP(hidden_dim, TARGET_DIM, hidden=128, num_layers=2)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero-initialize GRU hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        track_ctx: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step forward pass.

        Args:
            state:     (B, STATE_DIM)
            action:    (B, ACTION_DIM)
            track_ctx: (B, N_POINTS, 4)
            hidden:    (B, H) GRU hidden state

        Returns:
            new_hidden: (B, H) updated GRU hidden state
            delta_pred: (B, TARGET_DIM) predicted state delta (normalized)
        """
        h_s = self.state_enc(state)          # (B, 128)
        h_a = self.action_enc(action)        # (B, 32)
        h_t = self.track_enc(track_ctx)      # (B, track_out_dim)

        combined = torch.cat([h_s, h_a, h_t], dim=-1)  # (B, gru_input)
        new_hidden = self.gru_cell(combined, hidden)     # (B, H)
        delta_pred = self.pred_head(new_hidden)          # (B, TARGET_DIM)

        return new_hidden, delta_pred

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, run_config: 'RunConfig') -> 'WorldModel':
        """
        Construct a WorldModel from a RunConfig — the canonical way to
        rebuild architecture consistent with a saved checkpoint (see
        checkpointing.py). Requires run_config.num_materials to already
        be resolved (i.e. after the geometry stage has run).
        """
        if run_config.num_materials <= 0:
            raise ValueError(
                f'num_materials={run_config.num_materials} on RunConfig — '
                f'has the geometry stage run yet?'
            )
        return cls(
            num_materials=run_config.num_materials,
            hidden_dim=run_config.hidden,
            track_out_dim=run_config.track_out,
        )