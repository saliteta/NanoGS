from __future__ import annotations

import argparse
import math
import os
import numpy as np
from tqdm import tqdm
from utils.params import RunParams, CostParams
from utils.ply_utils import (
    read_ply,
    store_ply,
)
from utils.cost import full_cost_pairs
from utils.merge import merge_pairs


def knn_indices(means: np.ndarray, k: int) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:
        raise RuntimeError("This baseline requires scipy (scipy.spatial.cKDTree).") from e
    tree = cKDTree(means)
    _, idx = tree.query(means, k=k + 1, workers=-1)
    return idx[:, 1:]

def edge_costs(
    edges: np.ndarray,          # (M,2) int32, u<v
    mu: np.ndarray,
    sc: np.ndarray,
    q: np.ndarray,
    op: np.ndarray,
    sh: np.ndarray,
    cp,
    block_edges: int = 100_000,
) -> np.ndarray:
    """
    Compute symmetric costs w_e for each undirected edge once.
    Returns w: (M,) float32
    """
    M = edges.shape[0]
    w = np.empty((M,), dtype=np.float32)

    for e0 in tqdm(range(0, M, block_edges), desc="Edge costs"):
        e1 = min(M, e0 + block_edges)
        uv = edges[e0:e1]
        u = uv[:, 0]
        v = uv[:, 1]

        mu_u, sc_u, q_u, op_u = mu[u], sc[u], q[u], op[u]
        mu_v, sc_v, q_v, op_v = mu[v], sc[v], q[v], op[v]

        if sh.shape[1]:
            sh_u = sh[u]
            sh_v = sh[v]
        else:
            # keep shape consistent; full_cost_pairs expects (B,C) even if C=0
            sh_u = sh[u]
            sh_v = sh_u

        w[e0:e1] = full_cost_pairs(
            mu_u, sc_u, q_u, op_u, sh_u,
            mu_v, sc_v, q_v, op_v, sh_v,
            cp,
        ).astype(np.float32)

    return w

def knn_undirected_edges(nbr: np.ndarray) -> np.ndarray:
    """
    nbr: (N,k) int32 indices (directed kNN).
    Return edges: (M,2) int32 undirected edges with i<j, unique.
    Includes {i,j} if j in kNN(i) OR i in kNN(j) (union).
    """
    N, k = nbr.shape
    ii = np.repeat(np.arange(N, dtype=np.int32), k)
    jj = nbr.reshape(-1).astype(np.int32)

    u = np.minimum(ii, jj)
    v = np.maximum(ii, jj)

    # remove self edges if any
    mask = u != v
    u = u[mask]
    v = v[mask]

    edges = np.stack([u, v], axis=1)  # (N*k, 2) with u<v
    # unique undirected edges
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)

def greedy_pairs_from_edges(
    edges: np.ndarray,   # (M,2) int32, u<v
    w: np.ndarray,       # (M,) float32 costs
    N: int,
    P: int | None,       # how many pairs you want this pass
) -> np.ndarray:
    """
    Sort all edges by weight and greedily pick disjoint pairs.
    """
    if edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # filter invalid costs if any
    valid = np.isfinite(w)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32)

    idx = np.nonzero(valid)[0]
    order = idx[np.argsort(w[idx], kind="mergesort")]  # stable

    used = np.zeros(N, dtype=bool)
    pairs = []
    for ei in order:
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        if used[u] or used[v]:
            continue
        used[u] = True
        used[v] = True
        pairs.append((u, v))
        if P is not None and len(pairs) >= P:
            break

    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


def prune_by_opacity(mu, sc, q, op, sh, threshold=0.1):
    print("Opacity Mean:", np.mean(op), "Median:", np.median(op))
    threshold = min(threshold, np.median(op))
    print(f"Pruning splats with opacity below {threshold:.4f}")
    keep_idx = np.nonzero(op >= threshold)[0]
    print(f"Original count: {mu.shape[0]}, after opacity pruning: {keep_idx.shape[0]}")

    mu = mu[keep_idx]
    sc = sc[keep_idx]
    q  = q[keep_idx]
    op = op[keep_idx]
    if sh.shape[1]:
        sh = sh[keep_idx]
    return mu, sc, q, op, sh

def simplify(in_path: str, out_path: str, rp: RunParams, cp: CostParams) -> None:

    print(f"Loading PLY: {in_path}")
    hdr, mu, op, sc, q, sh, app_names = read_ply(in_path)
    N0 = int(mu.shape[0])
    print(f"Initial splats: {mu.shape[0]}")
    target = max(int(math.ceil(N0 * rp.ratio)), 1)
    print(f"Pruned splats: {N0}, target: {target}")
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
    op = np.clip(op, 0.0, 1.0).astype(np.float32)
    store_ply(out_path, hdr, mu, op, sc, q, sh, app_names)

    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ply",
        dest="ply",
        required=True,
        help="Input PLY file (raw 3DGS attributes)."
    )
    ap.add_argument(
        "-o", "--output",
        dest="output",
        default=None,
        help="Output PLY path. If omitted, auto-generated from input name and ratio."
    )
    ap.add_argument(
        "-r", "--ratio",
        type=float,
        default=0.5,
        help="Fraction of splats to keep, in (0,1). Example: 0.25 keeps 25%%."
    )

    ap.add_argument("--k", type=int, default=16, help="k for KNN candidates.")
    ap.add_argument("--merge_cap", type=float, default=0.5, help="Max merges per pass as ratio of original splat count (0.01–0.5).")
    ap.add_argument("--opacity_threshold", type=float, default=0.1, help="Prune splats with opacity below this threshold before merging.")

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

    rp = RunParams(
        ratio=args.ratio,
        merge_cap=merge_cap,
        k=args.k,
        opacity_threshold=args.opacity_threshold,
    )
    cp = CostParams(
        lam_geo=args.lam_geo,
        lam_sh=args.lam_sh,
    )

    simplify(args.ply, out_path, rp, cp)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
