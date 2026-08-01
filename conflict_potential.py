"""
机制 2 — Semantic Conflict Potential (语义冲突势)
=====================================================

Equation (3) in the paper:
  U(t) = Σ_{c ∈ C_act} w_c (1 − cos(μ_c^(t), μ_c^(t−Δ)))  +  γ · D_sym(P_t ‖ P_{t−Δ})

where:
  - The first term captures centroid semantic drift
  - The second term captures attention-mass redistribution via symmetric JS divergence

Then: Û(t) = max(0, U_{Δs}(t) − U_{Δl}(t))   ← dual-timescale detector

Adaptive EWMA threshold (Eq. 4):
  τ_t = Ū_t + κ·σ_t
  Ū_t = (1−ρ) Ū_{t−1} + ρ·Û(t)
  σ²_t = (1−ρ) σ²_{t−1} + ρ·(Û(t) − Ū_t)²

Defaults from paper: ρ_EWMA = 0.1, κ = 2.0, T_burn = 512
"""

from __future__ import annotations

__all__ = [
    "compute_centroid_drift",
    "compute_attention_divergence",
    "semantic_conflict_potential",
    "ConflictDetector",
]

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# 3.1  Centroid drift  (first term of Eq. 3)
# ---------------------------------------------------------------------------

def compute_centroid_drift(
    centroids_now: np.ndarray,      # [m, d]  current cluster centroids
    centroids_old: np.ndarray,      # [m, d]  centroids at t−Δ
    cluster_sizes: np.ndarray,      # [m]     |c| per cluster
) -> float:
    """
    Σ_c w_c (1 − cos(μ_c^(t), μ_c^(t−Δ)))
    where w_c ∝ s_c
    """
    m = len(cluster_sizes)
    if m == 0:
        return 0.0

    weights = cluster_sizes.astype(np.float64)
    total = weights.sum()
    if total == 0:
        weights = np.ones(m) / m
    else:
        weights /= total

    # cosine similarity: dot product (both should be L2-normalised)
    norms_now = np.linalg.norm(centroids_now, axis=1) + 1e-10
    norms_old = np.linalg.norm(centroids_old, axis=1) + 1e-10
    cos_sim = np.sum(centroids_now * centroids_old, axis=1) / (norms_now * norms_old)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    drift = np.sum(weights * (1.0 - cos_sim))
    return float(drift)


# ---------------------------------------------------------------------------
# 3.2  Attention-mass divergence  (second term of Eq. 3)
# ---------------------------------------------------------------------------

def compute_attention_divergence(
    attn_now: np.ndarray,    # [m]  P_t  (normalised attention mass per cluster)
    attn_old: np.ndarray,    # [m]  P_{t−Δ}
    eps: float = 1e-10,
) -> float:
    """
    Symmetric Jensen-Shannon divergence:  D_sym(P_t ‖ P_{t−Δ})

    JS(P, Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M)   where M = (P+Q)/2
    """
    p = np.asarray(attn_now, dtype=np.float64) + eps
    q = np.asarray(attn_old, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return float(0.5 * (kl_pm + kl_qm))


# ---------------------------------------------------------------------------
# 3.3  Combined conflict potential  —  Eq. (3)
# ---------------------------------------------------------------------------

def semantic_conflict_potential(
    centroids_now: np.ndarray,
    centroids_old: np.ndarray,
    cluster_sizes: np.ndarray,
    attn_now: np.ndarray,
    attn_old: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """
    U(t) = centroid_drift + γ · D_sym(P_t ‖ P_{t−Δ})
    """
    drift = compute_centroid_drift(centroids_now, centroids_old, cluster_sizes)
    div = compute_attention_divergence(attn_now, attn_old)
    return drift + gamma * div


# ---------------------------------------------------------------------------
# 3.4  ConflictDetector  — online boundary detection with dual-timescale EWMA
# ---------------------------------------------------------------------------

@dataclass
class ConflictDetector:
    """
    在线边界检测器。

    内部维护:
      - dual-timescale U buffer:  Û(t) = max(0, U_short − U_long)
      - EWMA statistics:  Ū_t, σ²_t
      - active episode id, refractory counter
    """

    rho: float = 0.1          # ρ_EWMA  — EWMA smoothing factor
    kappa: float = 2.0         # κ       — std-dev multiplier for threshold
    gamma: float = 1.0         # γ       — weight for attention-divergence term
    burn_in: int = 512         # T_burn
    refractory: int = 64        # T_refr  — minimum tokens between boundary triggers

    # state
    u_mean: float = field(default=0.0, init=False)
    u_var: float = field(default=1e-12, init=False)
    step_count: int = field(default=0, init=False)
    steps_since_trigger: int = field(default=0, init=False)
    episode_id: int = field(default=0, init=False)

    # dual-timescale buffers — pre-allocated as a fixed-size ring
    _u_short: float = 0.0
    _u_long: float = 0.0
    _u_history: list = field(default_factory=list)

    # snapshot storage for (centroids_old, attn_old) to pass
    # into the next call — the caller persists these between steps.
    _last_centroids: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_attn: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def step(
        self,
        centroids: np.ndarray,     # [m, d]
        cluster_sizes: np.ndarray, # [m]
        attn: np.ndarray,          # [m]
    ) -> bool:
        """
        Process one decode step.

        Returns:
          True  if Û(t) > τ_t  AND refractory period elapsed → episode boundary
          False otherwise
        """
        self.step_count += 1
        self.steps_since_trigger += 1

        # --- first call: store snapshot, no detection yet ---
        if self._last_centroids is None:
            self._last_centroids = centroids.copy()
            self._last_attn = attn.copy()
            return False

        # --- compute raw U(t) ---
        u_t = semantic_conflict_potential(
            centroids, self._last_centroids, cluster_sizes,
            attn, self._last_attn, gamma=self.gamma,
        )

        # update snapshot for next call
        self._last_centroids[:] = centroids
        self._last_attn[:] = attn

        # ---- dual-timescale  Û(t) = max(0, U_short − U_long) ----
        # simple enc: maintain two EWMA of U_t at different scales
        alpha_s = 0.3    # short window
        alpha_l = 0.01   # long  window
        self._u_short = alpha_s * u_t + (1.0 - alpha_s) * self._u_short
        self._u_long  = alpha_l * u_t + (1.0 - alpha_l) * self._u_long
        u_hat = max(0.0, self._u_short - self._u_long)

        # ---- EWMA threshold (Eq. 4) ----
        rho = self.rho
        self.u_mean = (1.0 - rho) * self.u_mean + rho * u_hat
        diff_sq = (u_hat - self.u_mean) ** 2
        self.u_var  = (1.0 - rho) * self.u_var  + rho * diff_sq
        tau = self.u_mean + self.kappa * np.sqrt(max(self.u_var, 1e-16))

        # ---- gate logic ----
        fired = False
        if self.step_count < self.burn_in:
            fired = False
        elif u_hat > tau and self.steps_since_trigger >= self.refractory:
            fired = True
            self.steps_since_trigger = 0
            self.episode_id += 1

        return fired

    @property
    def threshold(self) -> float:
        return self.u_mean + self.kappa * np.sqrt(max(self.u_var, 1e-16))
