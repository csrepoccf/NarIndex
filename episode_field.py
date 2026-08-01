"""
机制 1 — Episode Field (场景场)
=================================

Equation (1) in the paper:
  φ_i = [k̂_i ; λ·ψ(p_i)] ∈ R^{d+r}

where k̂_i = k_i / ‖k_i‖₂ is the L2-normalized stored key,
and ψ(p_i) is the low-frequency temporal phase embedding from RoPE bands.

Equation (2) — Temporal phase embedding:
  ω_j = θ^{ −2(j−1)/d },  j = 1,…,d/2   (RoPE base frequencies)
  ψ(p_i) = normalize( [sin(ω_1·p_i), cos(ω_1·p_i), …, sin(ω_r·p_i), cos(ω_r·p_i)] )

After embedding, spherical k-means on {φ_i} produces episode-coherent clusters.
"""

from __future__ import annotations

__all__ = [
    "temporal_phase_embedding",
    "build_episode_field",
    "spherical_kmeans_cluster",
    "EpisodeField",
]

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 2.1  Temporal phase embedding  —  Eq. (2)
# ---------------------------------------------------------------------------

def temporal_phase_embedding(
    positions: np.ndarray,   # [n_tokens]
    d: int,                  # key dimension (per-head)
    r: int = 8,              # number of low-frequency RoPE bands to keep
    theta: float = 1e6,      # RoPE base frequency Θ
) -> np.ndarray:             # [n_tokens, 2r]
    """
    从 RoPE 低频带提取时间相位向量。

    RoPE 频率: ω_j = θ^(−2(j−1)/d),  j = 1,…,d/2
    只取前 r 个最低频带 (j = 1,…,r), 每个带产生 sin 和 cos 两个值。
    """
    n = len(positions)
    p = positions.astype(np.float64)            # [n]
    bands = np.arange(1, r + 1, dtype=np.float64)  # [r]

    # ω_j = θ^(−2(j−1)/d)
    omegas = theta ** (-2.0 * (bands - 1) / d)       # [r]
    phases = np.outer(p, omegas)                       # [n, r]

    psi = np.empty((n, 2 * r), dtype=np.float32)
    psi[:, 0::2] = np.sin(phases)                     # sin channels
    psi[:, 1::2] = np.cos(phases)                     # cos channels

    # L2 normalize each row
    norms = np.linalg.norm(psi, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return psi / norms


# ---------------------------------------------------------------------------
# 2.2  Build episode field  —  Eq. (1)
# ---------------------------------------------------------------------------

def build_episode_field(
    keys: np.ndarray,          # [n_tokens, d]   stored keys (already RoPE'd)
    positions: np.ndarray,     # [n_tokens]       absolute positions
    lam: float = 0.8,
    r: int = 8,
    theta: float = 1e6,
) -> np.ndarray:               # [n_tokens, d + 2r]
    """
    Equation (1): φ_i = [ k̂_i  ;  λ·ψ(p_i) ]

    Parameters:
      keys:      raw stored keys k_i  (un-normalized, already had RoPE applied)
      positions: token positions
      lam:       trade-off λ between semantic and temporal information
      r:         number of low-frequency RoPE bands
      theta:     RoPE base frequency Θ
    """
    d = keys.shape[1]
    # L2-normalize keys  →  k̂_i
    k_hat = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-10)

    # temporal phase  →  ψ(p_i)
    psi = temporal_phase_embedding(positions, d, r=r, theta=theta)

    # concatenate  →  φ_i
    phi = np.concatenate([k_hat, lam * psi], axis=1)  # [n, d + 2r]
    return phi


# ---------------------------------------------------------------------------
# 2.3  Spherical k-means on episode field
# ---------------------------------------------------------------------------

def spherical_kmeans_cluster(
    phi: np.ndarray,
    m: int,
    n_iter: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherical k-means on the augmented space φ_i.
    Returns (labels, centroids).
    """
    rng = np.random.default_rng(seed)
    n, dim = phi.shape

    # initialise centroids: pick m random points and re-normalise
    idx = rng.choice(n, size=m, replace=False)
    centroids = phi[idx].copy()
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10

    labels = np.zeros(n, dtype=np.int32)

    for _ in range(n_iter):
        # ---- E-step: assign to nearest centroid (cosine similarity) ----
        sim = phi @ centroids.T                         # [n, m]
        new_labels = np.argmax(sim, axis=1)

        # ---- M-step: re-compute centroids as mean of assigned φ_i ----
        new_centroids = np.zeros_like(centroids)
        for c in range(m):
            mask = (new_labels == c)
            if mask.sum() == 0:
                # keep old centroid (degenerate cluster)
                new_centroids[c] = centroids[c]
            else:
                new_centroids[c] = phi[mask].mean(axis=0)
        new_centroids /= np.linalg.norm(new_centroids, axis=1, keepdims=True) + 1e-10

        if np.allclose(labels, new_labels):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    return labels, centroids


# ---------------------------------------------------------------------------
# 2.4  EpisodeField — high-level class
# ---------------------------------------------------------------------------

@dataclass
class EpisodeField:
    """
    管理 episode-field 索引的构建和查询。
    """

    d: int                              # key dimension per head
    r: int = 8                          # temporal-phase bands
    lam: float = 0.8                    # Eq. (1) λ
    theta: float = 1e6                  # RoPE base Θ
    m: int = 8                          # initial number of episodes (clusters)

    # internal state — set after build()
    phi_: Optional[np.ndarray] = None           # [n, d+2r]  augmented vectors
    labels_: Optional[np.ndarray] = None        # [n]         cluster ids
    centroids_: Optional[np.ndarray] = None     # [m, d+2r]   cluster centres
    cluster_episode_groups_: List[np.ndarray] = field(default_factory=list)

    def build(
        self,
        keys: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        构造 episode field 并聚类。
        返回 labels, 长度为 n。
        """
        self.phi_ = build_episode_field(
            keys, positions, lam=self.lam, r=self.r, theta=self.theta
        )
        self.labels_, self.centroids_ = spherical_kmeans_cluster(
            self.phi_, self.m
        )
        return self.labels_

    def get_cluster_purity(self, ground_truth_episodes: np.ndarray) -> float:
        """Calculate cluster purity given ground-truth episode labels (for evaluation)."""
        if self.labels_ is None:
            return 0.0
        purity = 0.0
        for c in range(self.m):
            mask = self.labels_ == c
            if mask.sum() == 0:
                continue
            best = np.bincount(ground_truth_episodes[mask], minlength=self.m).max()
            purity += best
        return purity / len(self.labels_)

    def get_mixing_rate(self, ground_truth_episodes: np.ndarray) -> float:
        """Fraction of conflict tokens (second-largest episode in each cluster)."""
        if self.labels_ is None:
            return 0.0
        total = 0.0
        for c in range(self.m):
            mask = self.labels_ == c
            n = mask.sum()
            if n < 2:
                continue
            counts = np.bincount(ground_truth_episodes[mask])
            if len(counts) < 2:
                continue
            sorted_counts = np.sort(counts)
            total += sorted_counts[-2]  # second-largest episode
        return total / len(self.labels_)

    @property
    def dispersion(self) -> float:
        """Average within-cluster variance (Term A in Theorem 1)."""
        if self.labels_ is None or self.centroids_ is None:
            return 0.0
        total = 0.0
        for c in range(self.m):
            mask = self.labels_ == c
            if mask.sum() < 2:
                continue
            cluster_points = self.phi_[mask]
            diffs = cluster_points - self.centroids_[c][np.newaxis, :]
            total += np.sum(diffs ** 2)
        return total / len(self.labels_)
