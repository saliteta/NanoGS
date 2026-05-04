from __future__ import annotations

import math
import numpy as np
import torch

from .params import CostParams
from .splat_utils import (
    quat_to_rotmat_batch,
    gauss_logpdf_diagrot_batch,
)


_LOG_2PI_POW_1P5 = math.log((2.0 * math.pi) ** 1.5)


def full_cost_pairs(
    mu_i: torch.Tensor,
    s_i: torch.Tensor,
    q_i: torch.Tensor,
    a_i: torch.Tensor,
    sh_i: torch.Tensor,
    mu_j: torch.Tensor,
    s_j: torch.Tensor,
    q_j: torch.Tensor,
    a_j: torch.Tensor,
    sh_j: torch.Tensor,
    cost: CostParams,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    n_mc = int(getattr(cost, "n_mc", 1))
    seed = int(getattr(cost, "seed", 0))
    eps_cov = float(getattr(cost, "eps_cov", 1e-8))

    device = mu_i.device
    dtype = mu_i.dtype

    R_i = quat_to_rotmat_batch(q_i)
    R_j = quat_to_rotmat_batch(q_j)
    Rt_i = R_i.transpose(1, 2)
    Rt_j = R_j.transpose(1, 2)

    v_i = s_i * s_i + eps_cov
    v_j = s_j * s_j + eps_cov

    invdiag_i = 1.0 / v_i.clamp_min(1e-30)
    invdiag_j = 1.0 / v_j.clamp_min(1e-30)

    logdet_i = torch.log(v_i.clamp_min(1e-30)).sum(dim=1)
    logdet_j = torch.log(v_j.clamp_min(1e-30)).sum(dim=1)

    # mixture weights (in log-space to avoid overflow with the (2pi)^1.5 prefactor)
    log_w_i = _LOG_2PI_POW_1P5 + torch.log(a_i.clamp_min(1e-30)) + torch.log(s_i.clamp_min(1e-30)).sum(dim=1)
    log_w_j = _LOG_2PI_POW_1P5 + torch.log(a_j.clamp_min(1e-30)) + torch.log(s_j.clamp_min(1e-30)).sum(dim=1)
    # pi = w_i / (w_i + w_j) = sigmoid(log_w_i - log_w_j)
    pi = torch.sigmoid(log_w_i - log_w_j).clamp(1e-12, 1.0 - 1e-12)
    log_pi = torch.log(pi)
    log_pj = torch.log(1.0 - pi)

    mu_m = pi[:, None] * mu_i + (1.0 - pi)[:, None] * mu_j

    di = mu_i - mu_m
    dj = mu_j - mu_m
    odi = di[:, :, None] * di[:, None, :]
    odj = dj[:, :, None] * dj[:, None, :]

    # Sig = R diag(v) R^T
    Sig_i = (R_i * v_i[:, None, :]) @ Rt_i
    Sig_j = (R_j * v_j[:, None, :]) @ Rt_j

    Sig_m = pi[:, None, None] * (Sig_i + odi) + (1.0 - pi)[:, None, None] * (Sig_j + odj)
    I = torch.eye(3, dtype=dtype, device=device).expand_as(Sig_m)
    Sig_m = 0.5 * (Sig_m + Sig_m.transpose(1, 2)) + eps_cov * I

    logdet_m = torch.linalg.slogdet(Sig_m).logabsdet

    k = 3.0
    log2pi = float(np.log(2.0 * np.pi))
    E_p_neglogq = 0.5 * (k * log2pi + logdet_m + k)

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    Z = torch.randn((n_mc, 3), dtype=dtype, device=device, generator=generator)

    std_i = torch.sqrt(v_i.clamp_min(0.0))
    std_j = torch.sqrt(v_j.clamp_min(0.0))

    Zi = Z[None, :, :] * std_i[:, None, :]
    Zj = Z[None, :, :] * std_j[:, None, :]

    x_i = mu_i[:, None, :] + torch.matmul(Zi, Rt_i)
    x_j = mu_j[:, None, :] + torch.matmul(Zj, Rt_j)

    logNi_on_i = gauss_logpdf_diagrot_batch(x_i, mu_i, R_i, invdiag_i, logdet_i)
    logNj_on_i = gauss_logpdf_diagrot_batch(x_i, mu_j, R_j, invdiag_j, logdet_j)
    logp_on_i = torch.logaddexp(log_pi[:, None] + logNi_on_i, log_pj[:, None] + logNj_on_i)
    Ei = logp_on_i.mean(dim=1)

    logNi_on_j = gauss_logpdf_diagrot_batch(x_j, mu_i, R_i, invdiag_i, logdet_i)
    logNj_on_j = gauss_logpdf_diagrot_batch(x_j, mu_j, R_j, invdiag_j, logdet_j)
    logp_on_j = torch.logaddexp(log_pi[:, None] + logNi_on_j, log_pj[:, None] + logNj_on_j)
    Ej = logp_on_j.mean(dim=1)

    E_p_logp = pi * Ei + (1.0 - pi) * Ej
    geo = E_p_logp + E_p_neglogq

    if sh_i.shape[1] == 0:
        c_sh = torch.zeros_like(geo)
    else:
        diff = sh_i - sh_j
        c_sh = (diff * diff).sum(dim=1)

    return cost.lam_geo * geo + cost.lam_sh * c_sh
