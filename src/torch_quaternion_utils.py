"""
Torch-native quaternion and rotation-vector math.

Mirrors quaternion_utils.py (numpy) but implemented in torch so it can
be used directly inside autograd-tracked training code (train.py) and
GPU-resident inference code (track_context.py) without ever crossing
back to numpy/CPU.

All quaternions are [W, X, Y, Z], Hamilton convention, unit-length.
"""

import torch


def torch_quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=eps)


def torch_quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def torch_quat_rotate(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vector v to world frame using q."""
    q = torch_quat_normalize(q)
    w = q[..., 0:1]
    u = q[..., 1:]
    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def torch_quat_rotate_inv(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate world-frame vector v into q's body frame (inverse rotation).

    Ported directly from quaternion_utils.quat_rotate_inv's formula
    (rather than re-derived) to guarantee identical numerics between the
    numpy training-data path and this torch inference-time path.
    """
    q = torch_quat_normalize(q)          # (B, 1, 4)
    q_w = q[..., 0:1]                    # (B, 1, 1)
    q_vec = q[..., 1:4]                  # (B, 1, 3)
    a = v * (2.0 * q_w * q_w - 1.0)      # (B, N_PTS, 3) * (B, 1, 1) -> OK
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0   # (B,1,3) x (B,N_PTS,3) broadcasts to (B,N_PTS,3)
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


def torch_quat_from_rvec(rvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors to quaternions [W, X, Y, Z]."""
    theta = torch.norm(rvec, dim=-1, keepdim=True)
    half = theta / 2.0
    safe_theta = torch.clamp(theta, min=1e-8)

    axis = torch.zeros_like(rvec)
    mask = (theta > 1e-8).squeeze(-1)
    axis[mask] = rvec[mask] / safe_theta[mask]

    qw = torch.cos(half)
    qxyz = torch.where(
        theta > 1e-8,
        axis * torch.sin(half),
        torch.zeros_like(rvec),
    )
    return torch.cat([qw, qxyz], dim=-1)