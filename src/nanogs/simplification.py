from __future__ import annotations

import argparse
import math
import os
import numpy as np
import torch
from tqdm import tqdm

from .utils.params import RunParams, CostParams
from .utils.ply_utils import read_ply, store_ply
from .utils.cost import full_cost_pairs
from .utils.merge import merge_pairs


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def knn_indices_faiss(means: torch.Tensor, k: int, use_ivf: bool = True,
                      nlist: int | None = None, nprobe: int = 16) -> torch.Tensor:
    """
    GPU kNN via FAISS. Defaults to IVF (approximate) for speed.
    Returns (N, k) int64 indices of the k nearest neighbors (self excluded).
    """
    import faiss

    N = means.shape[0]
    k_eff = min(k, max(1, N - 1))
    d = means.shape[1]
    means_np = np.ascontiguousarray(means.detach().cpu().numpy().astype(np.float32))
    res = faiss.StandardGpuResources()

    if use_ivf and N >= 50_000:
        # IVF: ~4*sqrt(N) cells is a typical recipe; cap at 65536.
        if nlist is None:
            nlist = min(65536, max(1024, int(4 * (N ** 0.5))))
        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.train(means_np)
        gpu_index.add(means_np)
        gpu_index.nprobe = nprobe
        _, I = gpu_index.search(means_np, k_eff + 1)
    else:
        gpu_index = faiss.GpuIndexFlatL2(res, d)
        gpu_index.add(means_np)
        _, I = gpu_index.search(means_np, k_eff + 1)
    I = torch.from_numpy(I.astype(np.int64))

    rows = torch.arange(N, dtype=torch.int64).unsqueeze(1)
    is_self = I == rows
    is_invalid = I < 0  # FAISS uses -1 for "no result"
    drop = is_self | is_invalid

    # Stable-sort: keep original (distance) order, push dropped entries to the end.
    pos = torch.arange(I.shape[1], dtype=torch.int64).expand_as(I)
    key = torch.where(drop, torch.full_like(pos, I.shape[1] + 1), pos)
    _, perm = torch.sort(key, dim=1, stable=True)
    I_sorted = torch.gather(I, 1, perm)
    nbr = I_sorted[:, :k_eff].clone()

    # If some row has fewer than k_eff valid neighbors (extreme duplicates),
    # backfill with row 0's last valid neighbor — guaranteed in-range.
    bad = (nbr < 0) | (nbr >= N)
    if bad.any():
        # find any valid neighbor per row
        valid_any = (~drop).int().argmax(dim=1)  # first valid column
        fallback = torch.gather(I, 1, valid_any.unsqueeze(1)).expand_as(nbr)
        nbr = torch.where(bad, fallback, nbr)

    return nbr.to(means.device)


def knn_indices_torch(means: torch.Tensor, k: int, chunk_bytes: int = 4 * 1024 * 1024 * 1024) -> torch.Tensor:
    """
    Fallback: brute-force chunked kNN on GPU using cdist + topk.
    """
    N = means.shape[0]
    k_eff = min(k, max(1, N - 1))
    bytes_per_row = 4 * N
    chunk = max(1, min(N, chunk_bytes // max(bytes_per_row, 1)))
    out = torch.empty((N, k_eff), dtype=torch.long, device=means.device)
    for c0 in tqdm(range(0, N, chunk), desc="kNN", leave=False):
        c1 = min(N, c0 + chunk)
        d = torch.cdist(means[c0:c1], means)
        rows = torch.arange(c0, c1, device=means.device).unsqueeze(1)
        d.scatter_(1, rows, float("inf"))
        _, idx = torch.topk(d, k_eff, dim=1, largest=False, sorted=True)
        out[c0:c1] = idx
    return out


def knn_indices(means: torch.Tensor, k: int) -> torch.Tensor:
    try:
        import faiss  # noqa: F401
        return knn_indices_faiss(means, k)
    except ImportError:
        return knn_indices_torch(means, k)


def knn_undirected_edges(nbr: torch.Tensor) -> torch.Tensor:
    """
    nbr: (N,k) int64 directed kNN indices.
    Returns (M,2) int64 undirected edges with u<v, unique, no self-edges.
    """
    N, k = nbr.shape
    ii = torch.arange(N, device=nbr.device).repeat_interleave(k)
    jj = nbr.reshape(-1)
    u = torch.minimum(ii, jj)
    v = torch.maximum(ii, jj)
    mask = u != v
    u = u[mask]
    v = v[mask]
    # encode pair as single int64 for unique
    key = u * np.int64(N) + v
    key_unique = torch.unique(key)
    u_out = key_unique // N
    v_out = key_unique % N
    return torch.stack([u_out, v_out], dim=1)


def edge_costs(
    edges: torch.Tensor,
    mu, sc, q, op, sh,
    cp: CostParams,
    block_edges: int = 200_000,
) -> torch.Tensor:
    M = edges.shape[0]
    w = torch.empty((M,), dtype=torch.float32, device=mu.device)
    has_sh = sh.shape[1] > 0
    gen = torch.Generator(device=mu.device)
    gen.manual_seed(int(getattr(cp, "seed", 0)))
    for e0 in tqdm(range(0, M, block_edges), desc="Edge costs", leave=False):
        e1 = min(M, e0 + block_edges)
        u = edges[e0:e1, 0]
        v = edges[e0:e1, 1]
        sh_u = sh[u] if has_sh else sh
        sh_v = sh[v] if has_sh else sh
        w[e0:e1] = full_cost_pairs(
            mu[u], sc[u], q[u], op[u], sh_u,
            mu[v], sc[v], q[v], op[v], sh_v,
            cp,
            generator=gen,
        )
    return w


try:
    from numba import njit

    @njit(cache=True)
    def _greedy_match_inner(edges_sorted: np.ndarray, used: np.ndarray, cap: int) -> np.ndarray:
        M = edges_sorted.shape[0]
        pairs = np.empty((M, 2), dtype=np.int64)
        n_pairs = 0
        for i in range(M):
            u = edges_sorted[i, 0]
            v = edges_sorted[i, 1]
            if used[u] or used[v]:
                continue
            used[u] = True
            used[v] = True
            pairs[n_pairs, 0] = u
            pairs[n_pairs, 1] = v
            n_pairs += 1
            if n_pairs >= cap:
                break
        return pairs[:n_pairs]
except ImportError:
    def _greedy_match_inner(edges_sorted, used, cap):
        M = edges_sorted.shape[0]
        pairs = np.empty((M, 2), dtype=np.int64)
        n_pairs = 0
        for i in range(M):
            u = int(edges_sorted[i, 0])
            v = int(edges_sorted[i, 1])
            if used[u] or used[v]:
                continue
            used[u] = True
            used[v] = True
            pairs[n_pairs, 0] = u
            pairs[n_pairs, 1] = v
            n_pairs += 1
            if n_pairs >= cap:
                break
        return pairs[:n_pairs]


def greedy_pairs_from_edges(edges: torch.Tensor, w: torch.Tensor, N: int, P: int | None) -> torch.Tensor:
    """
    Sort edges by weight on GPU, then greedily pick disjoint pairs (numba-JIT inner loop).
    """
    if edges.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=edges.device)

    valid = torch.isfinite(w)
    if not valid.any():
        return torch.zeros((0, 2), dtype=torch.long, device=edges.device)

    edges_v = edges[valid]
    w_v = w[valid]
    order = torch.argsort(w_v, stable=True)
    edges_sorted = np.ascontiguousarray(edges_v[order].cpu().numpy())

    used = np.zeros(N, dtype=np.bool_)
    cap = int(P) if P is not None else edges_sorted.shape[0]
    pairs = _greedy_match_inner(edges_sorted, used, cap)
    if pairs.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=edges.device)
    return torch.from_numpy(pairs).to(edges.device)


def prune_by_opacity(mu, sc, q, op, sh, threshold=0.1):
    op_np = op.detach().cpu().numpy() if isinstance(op, torch.Tensor) else op
    print("Opacity Mean:", float(np.mean(op_np)), "Median:", float(np.median(op_np)))
    threshold = min(threshold, float(np.median(op_np)))
    print(f"Pruning splats with opacity below {threshold:.4f}")
    keep_mask = op >= threshold
    keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
    print(f"Original count: {mu.shape[0]}, after opacity pruning: {keep_idx.shape[0]}")
    mu = mu[keep_idx]
    sc = sc[keep_idx]
    q = q[keep_idx]
    op = op[keep_idx]
    if sh.shape[1]:
        sh = sh[keep_idx]
    return mu, sc, q, op, sh


def simplify(in_path: str, out_path: str, rp: RunParams, cp: CostParams) -> None:
    device = _pick_device()
    print(f"Using device: {device}")

    print(f"Loading PLY: {in_path}")
    hdr, mu_np, op_np, sc_np, q_np, sh_np, app_names = read_ply(in_path)
    N0 = int(mu_np.shape[0])
    print(f"Initial splats: {N0}")
    target = max(int(math.ceil(N0 * rp.ratio)), 1)
    print(f"Target splats: {target}")

    mu = torch.from_numpy(mu_np).to(device)
    sc = torch.from_numpy(sc_np).to(device)
    q = torch.from_numpy(q_np).to(device)
    op = torch.from_numpy(op_np).to(device)
    sh = torch.from_numpy(sh_np).to(device)

    mu, sc, q, op, sh = prune_by_opacity(mu, sc, q, op, sh, rp.opacity_threshold)
    print(f"After opacity pruning, {mu.shape[0]} splats remain.")

    p_cap = max(1, int(rp.merge_cap * N0))

    iteration = 0
    while True:
        if mu.shape[0] <= target:
            break
        N = int(mu.shape[0])
        print(f"Pass {iteration + 1}: {N} splats")

        k_eff = min(max(1, rp.k), max(1, N - 1))
        nbr = knn_indices(mu, k=k_eff)
        edges = knn_undirected_edges(nbr)
        w = edge_costs(edges, mu, sc, q, op, sh, cp)

        merges_needed = N - target
        P = min(merges_needed, p_cap) if merges_needed > 0 else None
        pairs = greedy_pairs_from_edges(edges, w, N=N, P=P)
        print(f"  edges: {edges.shape[0]}, pairs: {pairs.shape[0]} (need {merges_needed})")

        mu, sc, q, op, sh = merge_pairs(mu, sc, q, op, sh, pairs)
        iteration += 1

    print(f"Final splats: {mu.shape[0]}")
    op = op.clamp(0.0, 1.0)

    mu_out = mu.detach().cpu().numpy()
    sc_out = sc.detach().cpu().numpy()
    q_out = q.detach().cpu().numpy()
    op_out = op.detach().cpu().numpy()
    sh_out = sh.detach().cpu().numpy()
    store_ply(out_path, hdr, mu_out, op_out, sc_out, q_out, sh_out, app_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", dest="ply", required=True, help="Input PLY file (raw 3DGS attributes).")
    ap.add_argument("-o", "--output", dest="output", default=None,
                    help="Output PLY path. If omitted, auto-generated from input name and ratio.")
    ap.add_argument("-r", "--ratio", type=float, default=0.5,
                    help="Fraction of splats to keep, in (0,1).")
    ap.add_argument("--k", type=int, default=16, help="k for KNN candidates.")
    ap.add_argument("--merge_cap", type=float, default=0.5,
                    help="Max merges per pass as ratio of original splat count (0.01-0.5).")
    ap.add_argument("--opacity_threshold", type=float, default=0.1,
                    help="Prune splats with opacity below this before merging.")
    ap.add_argument("--lam_geo", type=float, default=1.0)
    ap.add_argument("--lam_sh", type=float, default=1.0)

    args = ap.parse_args()
    if not (0.0 < args.ratio < 1.0):
        raise ValueError("--ratio must be in the open interval (0, 1).")
    merge_cap = max(0.01, min(0.5, float(args.merge_cap)))

    if args.output is not None:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.ply)
        if ext.lower() != ".ply":
            raise ValueError("Input file must have .ply extension.")
        ratio_tag = f"{args.ratio}".rstrip("0").rstrip(".")
        out_path = f"{base}_{ratio_tag}.ply"

    rp = RunParams(ratio=args.ratio, merge_cap=merge_cap, k=args.k,
                   opacity_threshold=args.opacity_threshold)
    cp = CostParams(lam_geo=args.lam_geo, lam_sh=args.lam_sh)

    simplify(args.ply, out_path, rp, cp)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
