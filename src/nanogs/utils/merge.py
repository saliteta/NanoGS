import math
import torch

from .splat_utils import (
    rotmat_to_quat_batch,
    sigma_from_scale_quat_batch,
)

# cusolver's batched syev fails on huge batches (~1M+ of 3x3 matrices).
# MAGMA handles arbitrary batch sizes and is also fast for tiny matrices.
if torch.cuda.is_available():
    try:
        torch.backends.cuda.preferred_linalg_library("magma")
    except Exception:
        pass


_LOG_2PI_POW_1P5 = math.log((2.0 * math.pi) ** 1.5)


def merge_pairs(mu, sc, q, op, sh, pairs):
    if pairs.shape[0] == 0:
        print("  No pairs to merge")
        return mu, sc, q, op, sh

    print(f"  Merging {pairs.shape[0]} pairs")
    i = pairs[:, 0]
    j = pairs[:, 1]
    has_sh = sh.shape[1] > 0
    sh_i = sh[i] if has_sh else sh
    sh_j = sh[j] if has_sh else sh

    mu_m, sc_m, q_m, op_m, sh_m = moment_matching(
        mu[i], sc[i], q[i], op[i], sh_i,
        mu[j], sc[j], q[j], op[j], sh_j,
    )

    used = torch.zeros(mu.shape[0], dtype=torch.bool, device=mu.device)
    used[i] = True
    used[j] = True
    keep_idx = (~used).nonzero(as_tuple=False).squeeze(1)

    mu2 = torch.cat([mu[keep_idx], mu_m], dim=0)
    sc2 = torch.cat([sc[keep_idx], sc_m], dim=0)
    q2 = torch.cat([q[keep_idx], q_m], dim=0)
    op2 = torch.cat([op[keep_idx], op_m], dim=0)
    sh2 = torch.cat([sh[keep_idx], sh_m], dim=0) if has_sh else sh

    return mu2, sc2, q2, op2, sh2


def moment_matching(
    mu_i, s_i, q_i, a_i, sh_i,
    mu_j, s_j, q_j, a_j, sh_j,
):
    """
    Inputs are ACTIVATED:
      - s: linear scales
      - a: alpha in (0, 1]
      - q: normalized quaternion
    """
    Sig_i = sigma_from_scale_quat_batch(s_i, q_i)
    Sig_j = sigma_from_scale_quat_batch(s_j, q_j)

    # work in log space to keep weights stable, then take softmax for normalized pi
    log_w_i = _LOG_2PI_POW_1P5 + torch.log(a_i.clamp_min(1e-30)) + torch.log(s_i.clamp_min(1e-30)).sum(dim=1)
    log_w_j = _LOG_2PI_POW_1P5 + torch.log(a_j.clamp_min(1e-30)) + torch.log(s_j.clamp_min(1e-30)).sum(dim=1)
    pi = torch.sigmoid(log_w_i - log_w_j).clamp(1e-12, 1.0 - 1e-12)  # w_i / (w_i + w_j)
    pj = 1.0 - pi

    mu = pi[:, None] * mu_i + pj[:, None] * mu_j

    di = mu_i - mu
    dj = mu_j - mu
    odi = di[:, :, None] * di[:, None, :]
    odj = dj[:, :, None] * dj[:, None, :]

    Sig = pi[:, None, None] * (Sig_i + odi) + pj[:, None, None] * (Sig_j + odj)
    I = torch.eye(3, dtype=Sig.dtype, device=Sig.device).expand_as(Sig)
    Sig = 0.5 * (Sig + Sig.transpose(1, 2)) + 1e-8 * I

    # Chunk eigh: cusolver's batched syevj has a batch-size limit; split big batches.
    EIGH_CHUNK = 200_000
    if Sig.shape[0] <= EIGH_CHUNK:
        evals, evecs = torch.linalg.eigh(Sig)
    else:
        ev_list, vc_list = [], []
        for c0 in range(0, Sig.shape[0], EIGH_CHUNK):
            ev, vc = torch.linalg.eigh(Sig[c0:c0 + EIGH_CHUNK])
            ev_list.append(ev)
            vc_list.append(vc)
        evals = torch.cat(ev_list, dim=0)
        evecs = torch.cat(vc_list, dim=0)
    evals = evals.clamp_min(1e-18)

    # sort descending
    order = torch.argsort(evals, dim=1, descending=True)
    evals = torch.gather(evals, 1, order)
    evecs = torch.gather(evecs, 2, order[:, None, :].expand(-1, 3, -1))

    # enforce right-handed
    detR = torch.linalg.det(evecs)
    flip = detR < 0
    if flip.any():
        evecs[flip, :, 2] = -evecs[flip, :, 2]

    scales = torch.sqrt(evals)
    quat = rotmat_to_quat_batch(evecs.contiguous())
    op = a_i + a_j - a_i * a_j

    if sh_i.shape[1] == 0:
        sh = sh_i
    else:
        sh = pi[:, None] * sh_i + pj[:, None] * sh_j

    return mu, scales, quat, op, sh
