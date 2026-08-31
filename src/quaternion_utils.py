"""
Quaternion and rotation-vector math for the Trackmania world model.

All quaternions are [W, X, Y, Z], Hamilton convention, unit-length.
All rotation vectors are 3-D axis-angle (direction = axis, norm = angle in radians).
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Quaternion algebra
# ──────────────────────────────────────────────────────────────────────

def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Hamilton product q ⊗ r.

    Args:
        q: (..., 4) quaternion(s) [W, X, Y, Z]
        r: (..., 4) quaternion(s) [W, X, Y, Z]

    Returns:
        (..., 4) product quaternion(s)
    """
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate [w, -x, -y, -z]."""
    out = np.empty_like(q)
    out[..., 0] = q[..., 0]
    out[..., 1] = -q[..., 1]
    out[..., 2] = -q[..., 2]
    out[..., 3] = -q[..., 3]
    return out


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Project onto the unit hypersphere."""
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return q / norms


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse quaternion.  For unit quaternions this equals the conjugate."""
    return quat_conjugate(q)


# ──────────────────────────────────────────────────────────────────────
# Rotation
# ──────────────────────────────────────────────────────────────────────

def quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vectors v by quaternion q (body → world).

    Equivalent to R(q) @ v but faster (no full matrix construction).

    Args:
        v: (..., 3) vectors in body frame
        q: (..., 4) unit quaternions [W, X, Y, Z]

    Returns:
        (..., 3) vectors in world frame
    """
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def quat_rotate_inv(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vectors v by the *inverse* of quaternion q (world → body).

    Equivalent to R(q)^T @ v but faster (no full matrix construction).

    Args:
        v: (..., 3) vectors in world frame
        q: (..., 4) unit quaternions [W, X, Y, Z]

    Returns:
        (..., 3) vectors in body frame
    """
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]

    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.sum(q_vec * v, axis=-1, keepdims=True)) * 2.0
    return a - b + c


# ──────────────────────────────────────────────────────────────────────
# Rotation-vector ↔ quaternion
# ──────────────────────────────────────────────────────────────────────

def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vectors to unit quaternions [W, X, Y, Z].

    Handles the zero-vector edge case (returns identity quaternion).
    """
    angle = np.linalg.norm(rvec, axis=-1, keepdims=True)  # (..., 1)
    half = angle / 2.0
    safe_angle = np.where(angle < 1e-10, 1.0, angle)
    s = np.sin(half) / safe_angle
    s = np.where(angle < 1e-8, 0.0, s)

    w = np.cos(half)
    xyz = rvec * s
    return np.concatenate([w, xyz], axis=-1)


# Alias for consistency with Dev 2's naming convention
quat_from_rvec = rvec_to_quat


def quat_to_rvec(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternions to rotation vectors.

    Always returns the shortest-arc representation (angle in [0, π]).
    Uses arctan2 for numerical stability across all quadrants.
    """
    q_pos = np.where(q[..., 0:1] < 0.0, -q, q)  # force w ≥ 0
    w = np.clip(q_pos[..., 0:1], -1.0, 1.0)
    xyz = q_pos[..., 1:4]
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm_xyz, w)  # (..., 1)

    safe_norm = np.where(norm_xyz < 1e-10, 1.0, norm_xyz)
    scale = angle / safe_norm
    scale = np.where(norm_xyz < 1e-10, 0.0, scale)

    return (xyz * scale).astype(np.float64)


def quat_relative(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """
    Relative rotation: q_rel such that q_to = q_rel ⊗ q_from.

    Computed as  q_to ⊗ q_from^{-1}.
    """
    return quat_multiply(q_to, quat_inverse(q_from))


# ──────────────────────────────────────────────────────────────────────
# Hemisphere fix
# ──────────────────────────────────────────────────────────────────────

def fix_quat_hemisphere(quat: np.ndarray) -> np.ndarray:
    """
    Fix quaternion sign ambiguity (q ≡ -q) so consecutive samples never
    have an artificial sign discontinuity.

    Sign continuity is enforced against the immediately preceding
    (already-fixed) frame — NOT a single static reference frame (e.g.
    frame 0, as a naive implementation might use). A static reference
    only works if the trajectory's orientation never drifts more than
    90° from the start pose; any trajectory that spins, does a U-turn,
    or loops (all normal in Trackmania) would otherwise have legitimate
    large-angle frames incorrectly flipped, while a real sign-flip
    artifact occurring after such a drift could be missed entirely.

    Implemented as a cumulative product of per-step sign corrections
    rather than a Python loop over frames — this is mathematically
    equivalent to: walk the sequence, and whenever frame i's raw dot
    product with the (already-corrected) frame i-1 is negative, flip
    frame i and everything's "current sign" going forward. Concretely,
    if c[i] is the cumulative correction sign applied to frame i, then
    requiring corrected consecutive dot products to be non-negative
    gives the recurrence c[i] = c[i-1] * sign(raw_quat[i] . raw_quat[i-1]),
    i.e. c = cumprod(sign(consecutive raw dot products)).
    """
    if quat.ndim == 1:
        return quat.copy()

    raw_dots = np.sum(quat[1:] * quat[:-1], axis=-1)            # (N-1,)
    step_sign = np.sign(raw_dots)
    step_sign = np.where(step_sign == 0.0, 1.0, step_sign)       # exact-zero dot: no flip
    cumulative_sign = np.concatenate([[1.0], np.cumprod(step_sign)])  # (N,)

    return quat * cumulative_sign[..., np.newaxis]
