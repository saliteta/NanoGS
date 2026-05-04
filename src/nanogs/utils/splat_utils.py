import numpy as np
import torch


def quat_normalize(q):
    if isinstance(q, torch.Tensor):
        n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
        return q / n
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return q / n


def quat_to_rotmat_batch(q_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q_wxyz.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = torch.empty((q_wxyz.shape[0], 3, 3), dtype=q_wxyz.dtype, device=q_wxyz.device)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def rotmat_to_quat_batch(R: torch.Tensor) -> torch.Tensor:
    M = R.shape[0]
    device = R.device
    dtype = R.dtype
    q = torch.empty((M, 4), dtype=dtype, device=device)

    m00 = R[:, 0, 0]
    m11 = R[:, 1, 1]
    m22 = R[:, 2, 2]
    tr = m00 + m11 + m22

    mask_tr = tr > 0
    mask_00 = (m00 > m11) & (m00 > m22) & (~mask_tr)
    mask_11 = (m11 > m22) & (~mask_tr) & (~mask_00)
    mask_22 = (~mask_tr) & (~mask_00) & (~mask_11)

    if mask_tr.any():
        Rt = R[mask_tr]
        S = torch.sqrt(tr[mask_tr] + 1.0) * 2.0
        q[mask_tr, 0] = 0.25 * S
        q[mask_tr, 1] = (Rt[:, 2, 1] - Rt[:, 1, 2]) / S
        q[mask_tr, 2] = (Rt[:, 0, 2] - Rt[:, 2, 0]) / S
        q[mask_tr, 3] = (Rt[:, 1, 0] - Rt[:, 0, 1]) / S

    if mask_00.any():
        Rt = R[mask_00]
        S = torch.sqrt(1.0 + Rt[:, 0, 0] - Rt[:, 1, 1] - Rt[:, 2, 2]) * 2.0
        q[mask_00, 0] = (Rt[:, 2, 1] - Rt[:, 1, 2]) / S
        q[mask_00, 1] = 0.25 * S
        q[mask_00, 2] = (Rt[:, 0, 1] + Rt[:, 1, 0]) / S
        q[mask_00, 3] = (Rt[:, 0, 2] + Rt[:, 2, 0]) / S

    if mask_11.any():
        Rt = R[mask_11]
        S = torch.sqrt(1.0 + Rt[:, 1, 1] - Rt[:, 0, 0] - Rt[:, 2, 2]) * 2.0
        q[mask_11, 0] = (Rt[:, 0, 2] - Rt[:, 2, 0]) / S
        q[mask_11, 1] = (Rt[:, 0, 1] + Rt[:, 1, 0]) / S
        q[mask_11, 2] = 0.25 * S
        q[mask_11, 3] = (Rt[:, 1, 2] + Rt[:, 2, 1]) / S

    if mask_22.any():
        Rt = R[mask_22]
        S = torch.sqrt(1.0 + Rt[:, 2, 2] - Rt[:, 0, 0] - Rt[:, 1, 1]) * 2.0
        q[mask_22, 0] = (Rt[:, 1, 0] - Rt[:, 0, 1]) / S
        q[mask_22, 1] = (Rt[:, 0, 2] + Rt[:, 2, 0]) / S
        q[mask_22, 2] = (Rt[:, 1, 2] + Rt[:, 2, 1]) / S
        q[mask_22, 3] = 0.25 * S

    return quat_normalize(q)


def sigma_from_scale_quat_batch(scales: torch.Tensor, q_wxyz: torch.Tensor) -> torch.Tensor:
    R = quat_to_rotmat_batch(q_wxyz)
    s2 = scales * scales
    Rd = R * s2[:, None, :]
    Sigma = Rd @ R.transpose(1, 2)
    return Sigma


def gauss_logpdf_diagrot_batch(
    x: torch.Tensor,        # [B,S,3]
    mu: torch.Tensor,       # [B,3]
    R: torch.Tensor,        # [B,3,3]  (Sigma = R diag(v) R^T)
    invdiag: torch.Tensor,  # [B,3]    (= 1/v)
    logdet: torch.Tensor,   # [B]
) -> torch.Tensor:
    k = 3.0
    log2pi = float(np.log(2.0 * np.pi))
    d = x - mu[:, None, :]
    y = torch.matmul(d, R)
    quad = ((y * y) * invdiag[:, None, :]).sum(dim=2)
    return -0.5 * (k * log2pi + logdet[:, None] + quad)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
