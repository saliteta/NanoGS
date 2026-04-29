import numpy as np

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return q / n

def quat_to_rotmat_batch(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty((q_wxyz.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - wz)
    R[:, 0, 2] = 2*(xz + wy)
    R[:, 1, 0] = 2*(xy + wz)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - wx)
    R[:, 2, 0] = 2*(xz - wy)
    R[:, 2, 1] = 2*(yz + wx)
    R[:, 2, 2] = 1 - 2*(xx + yy)
    return R

def rotmat_to_quat_batch(R: np.ndarray) -> np.ndarray:
    M = R.shape[0]
    q = np.empty((M, 4), dtype=np.float32)

    m00 = R[:, 0, 0]; m11 = R[:, 1, 1]; m22 = R[:, 2, 2]
    tr = m00 + m11 + m22

    mask_tr = tr > 0
    mask_00 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & (~mask_tr)
    mask_11 = (R[:, 1, 1] > R[:, 2, 2]) & (~mask_tr) & (~mask_00)
    mask_22 = (~mask_tr) & (~mask_00) & (~mask_11)

    if np.any(mask_tr):
        S = np.sqrt(tr[mask_tr] + 1.0) * 2.0
        q[mask_tr, 0] = 0.25 * S
        q[mask_tr, 1] = (R[mask_tr, 2, 1] - R[mask_tr, 1, 2]) / S
        q[mask_tr, 2] = (R[mask_tr, 0, 2] - R[mask_tr, 2, 0]) / S
        q[mask_tr, 3] = (R[mask_tr, 1, 0] - R[mask_tr, 0, 1]) / S

    if np.any(mask_00):
        S = np.sqrt(1.0 + R[mask_00, 0, 0] - R[mask_00, 1, 1] - R[mask_00, 2, 2]) * 2.0
        q[mask_00, 0] = (R[mask_00, 2, 1] - R[mask_00, 1, 2]) / S
        q[mask_00, 1] = 0.25 * S
        q[mask_00, 2] = (R[mask_00, 0, 1] + R[mask_00, 1, 0]) / S
        q[mask_00, 3] = (R[mask_00, 0, 2] + R[mask_00, 2, 0]) / S

    if np.any(mask_11):
        S = np.sqrt(1.0 + R[mask_11, 1, 1] - R[mask_11, 0, 0] - R[mask_11, 2, 2]) * 2.0
        q[mask_11, 0] = (R[mask_11, 0, 2] - R[mask_11, 2, 0]) / S
        q[mask_11, 1] = (R[mask_11, 0, 1] + R[mask_11, 1, 0]) / S
        q[mask_11, 2] = 0.25 * S
        q[mask_11, 3] = (R[mask_11, 1, 2] + R[mask_11, 2, 1]) / S

    if np.any(mask_22):
        S = np.sqrt(1.0 + R[mask_22, 2, 2] - R[mask_22, 0, 0] - R[mask_22, 1, 1]) * 2.0
        q[mask_22, 0] = (R[mask_22, 1, 0] - R[mask_22, 0, 1]) / S
        q[mask_22, 1] = (R[mask_22, 0, 2] + R[mask_22, 2, 0]) / S
        q[mask_22, 2] = (R[mask_22, 1, 2] + R[mask_22, 2, 1]) / S
        q[mask_22, 3] = 0.25 * S

    return quat_normalize(q)

def sigma_from_scale_quat_batch(scales: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat_batch(q_wxyz)
    s2 = (scales * scales).astype(np.float32)
    Rd = R * s2[:, None, :]
    Sigma = Rd @ np.transpose(R, (0, 2, 1))
    return Sigma

def batch_inv_3x3(A: np.ndarray) -> np.ndarray:
    a00 = A[:, 0, 0]; a01 = A[:, 0, 1]; a02 = A[:, 0, 2]
    a10 = A[:, 1, 0]; a11 = A[:, 1, 1]; a12 = A[:, 1, 2]
    a20 = A[:, 2, 0]; a21 = A[:, 2, 1]; a22 = A[:, 2, 2]

    c00 = a11*a22 - a12*a21
    c01 = a02*a21 - a01*a22
    c02 = a01*a12 - a02*a11

    c10 = a12*a20 - a10*a22
    c11 = a00*a22 - a02*a20
    c12 = a02*a10 - a00*a12

    c20 = a10*a21 - a11*a20
    c21 = a01*a20 - a00*a21
    c22 = a00*a11 - a01*a10

    det = a00*c00 + a01*c10 + a02*c20
    det = np.where(np.abs(det) < 1e-18, 1e-18, det)
    inv_det = 1.0 / det

    invA = np.empty_like(A)
    invA[:, 0, 0] = c00 * inv_det
    invA[:, 0, 1] = c01 * inv_det
    invA[:, 0, 2] = c02 * inv_det
    invA[:, 1, 0] = c10 * inv_det
    invA[:, 1, 1] = c11 * inv_det
    invA[:, 1, 2] = c12 * inv_det
    invA[:, 2, 0] = c20 * inv_det
    invA[:, 2, 1] = c21 * inv_det
    invA[:, 2, 2] = c22 * inv_det
    return invA

def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, 1e-12))

def det_3x3(A: np.ndarray) -> np.ndarray:
    a00 = A[:, 0, 0]; a01 = A[:, 0, 1]; a02 = A[:, 0, 2]
    a10 = A[:, 1, 0]; a11 = A[:, 1, 1]; a12 = A[:, 1, 2]
    a20 = A[:, 2, 0]; a21 = A[:, 2, 1]; a22 = A[:, 2, 2]
    return (a00*(a11*a22 - a12*a21)
          - a01*(a10*a22 - a12*a20)
          + a02*(a10*a21 - a11*a20)).astype(np.float32)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def gauss_logpdf_diagrot_batch(
    x: np.ndarray,          # [B,S,3]
    mu: np.ndarray,         # [B,3]
    R: np.ndarray,          # [B,3,3]  (Sigma = R diag(v) R^T)
    invdiag: np.ndarray,    # [B,3]    (= 1/v)
    logdet: np.ndarray,     # [B]
) -> np.ndarray:
    """
    log N(x | mu, Sigma) where Sigma = R diag(v) R^T.
    Uses rotated-diagonal quadratic: quad = sum_k ( ( (x-mu) @ R )_k^2 * invdiag_k )
    """
    k = 3.0
    log2pi = np.log(2.0 * np.pi).astype(np.float32)

    d = x - mu[:, None, :]  # [B,S,3]
    # y = d @ R  (coordinates in eigenframe)
    y = np.matmul(d, R)     # [B,S,3]
    quad = np.sum((y * y) * invdiag[:, None, :], axis=2)  # [B,S]
    return -0.5 * (k * log2pi + logdet[:, None] + quad)