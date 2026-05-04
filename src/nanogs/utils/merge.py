import numpy as np
from .splat_utils import (
    rotmat_to_quat_batch,
    sigma_from_scale_quat_batch,
)

def merge_pairs(mu, sc, q, op, sh, pairs):
        if pairs.shape[0] == 0:
            print("  No pairs to merge")
            return mu, sc, q, op, sh

        print(f"  Merging {pairs.shape[0]} pairs")
        i = pairs[:, 0]
        j = pairs[:, 1]
        mu_m, sc_m, q_m, op_m, sh_m = moment_matching(
            mu[i], sc[i], q[i], op[i], sh[i],
            mu[j], sc[j], q[j], op[j], sh[j],
        )

        used = np.zeros(mu.shape[0], dtype=bool)
        used[i] = True
        used[j] = True
        keep_idx = np.nonzero(~used)[0]

        mu2 = np.concatenate([mu[keep_idx], mu_m], axis=0)
        sc2 = np.concatenate([sc[keep_idx], sc_m], axis=0)
        q2 = np.concatenate([q[keep_idx], q_m], axis=0)
        op2 = np.concatenate([op[keep_idx], op_m], axis=0)
        sh2 = np.concatenate([sh[keep_idx], sh_m], axis=0) if sh.shape[1] else sh

        return mu2, sc2, q2, op2, sh2

def moment_matching(
    mu_i: np.ndarray, s_i: np.ndarray, q_i: np.ndarray, a_i: np.ndarray, sh_i: np.ndarray,
    mu_j: np.ndarray, s_j: np.ndarray, q_j: np.ndarray, a_j: np.ndarray, sh_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs are ACTIVATED:
      - s: linear scales
      - a: alpha in above 0, clip at the place where we are saving
      - q: normalized quaternion
    Outputs are ACTIVATED (caller converts to RAW for writing).
    """
    Sig_i = sigma_from_scale_quat_batch(s_i, q_i)
    Sig_j = sigma_from_scale_quat_batch(s_j, q_j)

    w_i = (2 * np.pi) ** 1.5 * a_i * np.prod(s_i, axis=1).astype(np.float32) + np.float32(1e-12) 
    w_j = (2 * np.pi) ** 1.5 * a_j * np.prod(s_j, axis=1).astype(np.float32) + np.float32(1e-12) 

    W = np.maximum(w_i + w_j, 1e-12)

    mu = (w_i[:, None] * mu_i + w_j[:, None] * mu_j) / W[:, None]

    di = (mu_i - mu).astype(np.float32)
    dj = (mu_j - mu).astype(np.float32)
    odi = di[:, :, None] * di[:, None, :]
    odj = dj[:, :, None] * dj[:, None, :]

    Sig = (w_i[:, None, None] * (Sig_i + odi) + w_j[:, None, None] * (Sig_j + odj)) / W[:, None, None]
    Sig = 0.5 * (Sig + np.transpose(Sig, (0, 2, 1))) + np.eye(3, dtype=np.float32)[None, :, :] * 1e-8

    evals, evecs = np.linalg.eigh(Sig)
    evals = np.maximum(evals, 1e-18).astype(np.float32)

    # sort descending
    order = np.argsort(evals, axis=1)[:, ::-1]
    evals = np.take_along_axis(evals, order, axis=1)
    evecs = np.take_along_axis(evecs, order[:, None, :], axis=2)

    # enforce right-handed
    detR = np.linalg.det(evecs).astype(np.float32)
    flip = detR < 0
    if np.any(flip):
        evecs[flip, :, 2] *= -1.0

    scales = np.sqrt(evals).astype(np.float32)
    quat = rotmat_to_quat_batch(evecs.astype(np.float32))
    op = a_i + a_j - a_i * a_j

    if sh_i.shape[1] == 0:
        sh = sh_i
    else:
        sh = (w_i[:, None] * sh_i + w_j[:, None] * sh_j) / W[:, None]

    return mu.astype(np.float32), scales, quat, op, sh.astype(np.float32) if sh_i.shape[1] else sh_i

