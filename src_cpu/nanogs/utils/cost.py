from __future__ import annotations

import numpy as np


from .params import CostParams
from .splat_utils import (
    sigma_from_scale_quat_batch,
    batch_inv_3x3,
    safe_log,
    quat_to_rotmat_batch,
    gauss_logpdf_diagrot_batch,
)


def full_cost_pairs(
    mu_i: np.ndarray,
    s_i: np.ndarray,
    q_i: np.ndarray,
    a_i: np.ndarray,
    sh_i: np.ndarray,
    mu_j: np.ndarray,
    s_j: np.ndarray,
    q_j: np.ndarray,
    a_j: np.ndarray,
    sh_j: np.ndarray,
    cost: CostParams,
) -> np.ndarray:
    # knobs
    n_mc = int(getattr(cost, "n_mc", 1))
    seed = int(getattr(cost, "seed", 0))
    eps_cov = float(getattr(cost, "eps_cov", 1e-8))  # scalar variance jitter

    # ---- rotations ----
    R_i = quat_to_rotmat_batch(q_i)  # [B,3,3]
    R_j = quat_to_rotmat_batch(q_j)

    Rt_i = np.transpose(R_i, (0, 2, 1))
    Rt_j = np.transpose(R_j, (0, 2, 1))

    # ---- variances (rotation-invariant scalar jitter) ----
    # v = s^2 + eps, ensures SPD without matrix jitter loops
    v_i = (s_i.astype(np.float32) * s_i.astype(np.float32)) + np.float32(eps_cov)
    v_j = (s_j.astype(np.float32) * s_j.astype(np.float32)) + np.float32(eps_cov)

    invdiag_i = 1.0 / np.maximum(v_i, 1e-30).astype(np.float32)
    invdiag_j = 1.0 / np.maximum(v_j, 1e-30).astype(np.float32)

    logdet_i = np.sum(np.log(np.maximum(v_i, 1e-30)), axis=1).astype(np.float32)
    logdet_j = np.sum(np.log(np.maximum(v_j, 1e-30)), axis=1).astype(np.float32)

    # ---- mixture weights ----
    w_i = (2 * np.pi) ** 1.5 * a_i.astype(np.float32) * np.prod(s_i, axis=1).astype(np.float32) + np.float32(1e-12)          
    w_j = (2 * np.pi) ** 1.5 * a_j.astype(np.float32) * np.prod(s_j, axis=1).astype(np.float32) + np.float32(1e-12) 
    W = w_i + w_j
    W_safe = np.where(W > 0.0, W, 1.0).astype(np.float32)

    pi = (w_i / W_safe).astype(np.float32)
    pi = np.clip(pi, 1e-12, 1.0 - 1e-12).astype(np.float32)
    log_pi = np.log(pi).astype(np.float32)
    log_pj = np.log(1.0 - pi).astype(np.float32)

    # ---- moment-matched merge covariance (needs full Sig_i/Sig_j) ----
    mu_i32 = mu_i.astype(np.float32, copy=False)
    mu_j32 = mu_j.astype(np.float32, copy=False)

    mu_m = pi[:, None] * mu_i32 + (1.0 - pi)[:, None] * mu_j32

    di = mu_i32 - mu_m
    dj = mu_j32 - mu_m
    odi = di[:, :, None] * di[:, None, :]
    odj = dj[:, :, None] * dj[:, None, :]

    # Build full Sig from R diag(v) R^T (scale columns of R by v, then @ R^T)
    Sig_i = np.matmul(R_i * v_i[:, None, :], Rt_i)  # [B,3,3]
    Sig_j = np.matmul(R_j * v_j[:, None, :], Rt_j)

    Sig_m = pi[:, None, None] * (Sig_i + odi) + (1.0 - pi)[:, None, None] * (Sig_j + odj)
    # cheap SPD-ish stabilize for Sig_m only (sym + eps I)
    I = np.eye(3, dtype=np.float32)[None, :, :]
    Sig_m = 0.5 * (Sig_m + np.transpose(Sig_m, (0, 2, 1))) + np.float32(eps_cov) * I

    _, logdet_m = np.linalg.slogdet(Sig_m)
    logdet_m = logdet_m.astype(np.float32)

    # ---- KL(p_mix || q_merge) ----
    k = 3.0
    log2pi = np.log(2.0 * np.pi).astype(np.float32)
    E_p_neglogq = 0.5 * (k * log2pi + logdet_m + k).astype(np.float32)

    # deterministic MC samples (shared across pairs)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_mc, 3)).astype(np.float32)  # [S,3]

    # sample x = mu + (Z * std) @ R^T
    std_i = np.sqrt(np.maximum(v_i, 0.0)).astype(np.float32)
    std_j = np.sqrt(np.maximum(v_j, 0.0)).astype(np.float32)

    Zi = (Z[None, :, :] * std_i[:, None, :])  # [B,S,3]
    Zj = (Z[None, :, :] * std_j[:, None, :])  # [B,S,3]

    x_i = mu_i32[:, None, :] + np.matmul(Zi, Rt_i)  # [B,S,3]
    x_j = mu_j32[:, None, :] + np.matmul(Zj, Rt_j)

    # logpdfs using rotated-diagonal form (no inv, no cholesky)
    logNi_on_i = gauss_logpdf_diagrot_batch(x_i, mu_i32, R_i, invdiag_i, logdet_i)
    logNj_on_i = gauss_logpdf_diagrot_batch(x_i, mu_j32, R_j, invdiag_j, logdet_j)
    logp_on_i = np.logaddexp(log_pi[:, None] + logNi_on_i, log_pj[:, None] + logNj_on_i)
    Ei = np.mean(logp_on_i, axis=1).astype(np.float32)

    logNi_on_j = gauss_logpdf_diagrot_batch(x_j, mu_i32, R_i, invdiag_i, logdet_i)
    logNj_on_j = gauss_logpdf_diagrot_batch(x_j, mu_j32, R_j, invdiag_j, logdet_j)
    logp_on_j = np.logaddexp(log_pi[:, None] + logNi_on_j, log_pj[:, None] + logNj_on_j)
    Ej = np.mean(logp_on_j, axis=1).astype(np.float32)

    E_p_logp = (pi * Ei + (1.0 - pi) * Ej).astype(np.float32)
    KL_mix_to_merge = (E_p_logp + E_p_neglogq).astype(np.float32)

    geo = KL_mix_to_merge


    # ---- SH L2 ----
    if sh_i.shape[1] == 0:
        c_sh = np.zeros_like(geo, dtype=np.float32)
    else:
        diff = sh_i.astype(np.float32) - sh_j.astype(np.float32)
        c_sh = np.sum(diff * diff, axis=1).astype(np.float32)

    out = (cost.lam_geo * geo + cost.lam_sh * c_sh).astype(np.float32)
    return out

def full_cost_pairs_ij(
    mu_i: np.ndarray,
    s_i: np.ndarray,
    q_i: np.ndarray,
    a_i: np.ndarray,
    sh_i: np.ndarray,
    mu_j: np.ndarray,
    s_j: np.ndarray,
    q_j: np.ndarray,
    a_j: np.ndarray,
    sh_j: np.ndarray,
    cost: CostParams,
) -> np.ndarray:
    # knobs
    eps_cov = float(getattr(cost, "eps_cov", 1e-8))  # scalar variance jitter

    # ---- rotations ----
    R_i = quat_to_rotmat_batch(q_i).astype(np.float32)  # [B,3,3] world->local
    R_j = quat_to_rotmat_batch(q_j).astype(np.float32)

    Rt_i = np.transpose(R_i, (0, 2, 1))
    Rt_j = np.transpose(R_j, (0, 2, 1))

    # ---- variances (scalar jitter) ----
    s_i32 = s_i.astype(np.float32, copy=False)
    s_j32 = s_j.astype(np.float32, copy=False)

    v_i = s_i32 * s_i32 + np.float32(eps_cov)  # [B,3]
    v_j = s_j32 * s_j32 + np.float32(eps_cov)

    invdiag_i = (1.0 / np.maximum(v_i, 1e-30)).astype(np.float32)
    invdiag_j = (1.0 / np.maximum(v_j, 1e-30)).astype(np.float32)

    logdet_i = np.sum(np.log(np.maximum(v_i, 1e-30)), axis=1).astype(np.float32)  # [B]
    logdet_j = np.sum(np.log(np.maximum(v_j, 1e-30)), axis=1).astype(np.float32)

    mu_i32 = mu_i.astype(np.float32, copy=False)
    mu_j32 = mu_j.astype(np.float32, copy=False)

    # delta in world
    d_ij = (mu_i32 - mu_j32).astype(np.float32)  # [B,3]
    d_ji = -d_ij

    k = np.float32(3.0)

    # ---------- KL(i || j) ----------
    # mean quadratic: (mu_j - mu_i)^T Sigma_j^{-1} (mu_j - mu_i)
    # with R_j world->local: local coords = d @ R_j
    d_ij_in_j = np.einsum("bi,bij->bj", d_ij, R_j)  # [B,3]
    quad_ij = np.sum(d_ij_in_j * d_ij_in_j * invdiag_j, axis=1).astype(np.float32)

    # trace term: tr(Sigma_j^{-1} Sigma_i)
    # Let M = R_j^T R_i (maps i-local axes into j-local axes in terms of world rotations)
    # In j-local frame: R_j^T Sigma_i R_j = M diag(v_i) M^T
    # diag of that is sum_k M[a,k]^2 * v_i[k]
    M_ji = np.matmul(Rt_j, R_i).astype(np.float32)  # [B,3,3]
    diagC_ij = np.sum((M_ji * M_ji) * v_i[:, None, :], axis=2).astype(np.float32)  # [B,3]
    tr_ij = np.sum(invdiag_j * diagC_ij, axis=1).astype(np.float32)

    KL_ij = 0.5 * ((logdet_j - logdet_i) - k + tr_ij + quad_ij).astype(np.float32)

    # ---------- KL(j || i) ----------
    d_ji_in_i = np.einsum("bi,bij->bj", d_ji, R_i)  # [B,3]
    quad_ji = np.sum(d_ji_in_i * d_ji_in_i * invdiag_i, axis=1).astype(np.float32)

    M_ij = np.matmul(Rt_i, R_j).astype(np.float32)  # [B,3,3]
    diagC_ji = np.sum((M_ij * M_ij) * v_j[:, None, :], axis=2).astype(np.float32)
    tr_ji = np.sum(invdiag_i * diagC_ji, axis=1).astype(np.float32)

    KL_ji = 0.5 * ((logdet_i - logdet_j) - k + tr_ji + quad_ji).astype(np.float32)

    # symmetrized KL (Jeffreys)
    geo = 0.5 * (KL_ij + KL_ji).astype(np.float32)

    # ---- SH L2 ----
    if sh_i.shape[1] == 0:
        c_sh = np.zeros_like(geo, dtype=np.float32)
    else:
        diff = sh_i.astype(np.float32) - sh_j.astype(np.float32)
        c_sh = np.sum(diff * diff, axis=1).astype(np.float32)

    out = (cost.lam_geo * geo + cost.lam_sh * c_sh).astype(np.float32)
    return out
