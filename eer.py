"""
机制 3 — Episode-Exclusive Retrieval (EER, 场景独占检索)
==========================================================

Equation (5) — Retrieval score:
  s_c = q_t^⊤ μ_c  +  η·log a_c^rec  +  log π(c, t)

  where  a_c^rec = exp(−δ·Δt_c)            recency factor
         log π(c, t) = −ρ_EER · d_episode(c, e_active)

Equation (6) — Softmax injection:
  α̃_{t,i} = exp( (q_t^⊤ k_i + log π(c_i, t)) / √d ) / Z_t

Default:  η = 0.15,  ρ_EER = 1.0
"""

from __future__ import annotations

__all__ = [
    "compute_retrieval_scores",
    "episode_softmax_bias",
    "EpisodeExclusiveRetrieval",
]

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 4.1  Episode distance between two clusters
#      d_episode(c, e_active) = |idx(e_c) − idx(e_active)|
# ---------------------------------------------------------------------------

def _episode_distance(
    cluster_episode_tags: np.ndarray,   # [m]  episode tag per cluster
    active_episode: int,
) -> np.ndarray:                        # [m]  {0, 1, 2, …}
    return np.abs(cluster_episode_tags.astype(np.int64) - active_episode)


# ---------------------------------------------------------------------------
# 4.2  Compute retrieval scores  —  Eq. (5)
# ---------------------------------------------------------------------------

def compute_retrieval_scores(
    query: np.ndarray,                   # [d]                  query at time t
    centroids: np.ndarray,               # [m, d]               key-space centroids μ_c
    cluster_episode_tags: np.ndarray,    # [m]                  e_c per cluster
    active_episode: int,                 # current active e
    last_access: np.ndarray,             # [m]                  step last accessed
    current_step: int,
    rho_eer: float = 1.0,                # ρ_EER
    eta: float = 0.15,                   # recency weight η
    delta_rec: float = 0.001,            # δ  (per-token recency decay)
    temperature: float = 1.0,            # optional softmax temperature
) -> np.ndarray:                         # [m]   s_c
    """
    Equation (5):  s_c = q^⊤ μ_c  +  η·log a_c^rec  +  log π(c, t)
    """

    # --- semantic score ---
    sim = query @ centroids.T                              # [m]

    # --- recency factor ---
    dt = np.asarray(current_step - last_access, dtype=np.float64)
    log_rec = -delta_rec * np.maximum(dt, 0)                # [m]

    # --- episode-compatibility bias ---
    d_epi = _episode_distance(cluster_episode_tags, active_episode)   # [m]
    log_pi = -rho_eer * d_epi.astype(np.float64)                      # [m]

    scores = (sim + eta * log_rec + log_pi) / max(temperature, 1e-10)
    return scores


# ---------------------------------------------------------------------------
# 4.3  Softmax with episode bias  —  Eq. (6)
# ---------------------------------------------------------------------------

def episode_softmax_bias(
    query: np.ndarray,                   # [d]
    keys: np.ndarray,                    # [n, d]
    cluster_ids: np.ndarray,            # [n]    which cluster each token belongs to
    cluster_episode_tags: np.ndarray,   # [m]    episode tag per cluster
    active_episode: int,
    rho_eer: float = 1.0,
    d: Optional[float] = None,
) -> np.ndarray:                        # [n]  α̃_{t,i}
    """
    Equation (6):
      α̃_{t,i} = exp( (q_t^⊤ k_i + log π(c_i, t)) / √d ) / Z_t

    This is the per-token version — used for precise attention over
    the retrieval zone.  The partition function Z_t sums over all tokens.
    """
    dim = d or keys.shape[1]
    logits = query @ keys.T                        # [n]

    # episode bias per token
    epi_tags = cluster_episode_tags[cluster_ids]    # [n]
    d_epi = np.abs(epi_tags.astype(np.int64) - active_episode)
    bias = -rho_eer * d_epi.astype(np.float64)      # [n]

    scores = (logits + bias) / np.sqrt(dim)
    scores -= scores.max()                          # stability
    probs = np.exp(scores)
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# 4.4  EER class  — top-r retrieval with episode filtering
# ---------------------------------------------------------------------------

@dataclass
class EpisodeExclusiveRetrieval:
    """
    Episode-Exclusive Retrieval — injects episode-compatibility bias
    into partition scoring before top-r selection.
    """

    rho_eer: float = 1.0          # ρ_EER
    eta: float = 0.15             # recency weight
    delta_rec: float = 0.001      # per-token recency decay
    r: int = 64                   # retrieval budget (top-r)

    # dynamic state
    _step: int = field(default=0, init=False)
    _last_access: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def select_retrieval_zone(
        self,
        query: np.ndarray,                   # [d]
        centroids: np.ndarray,               # [m, d]
        cluster_episode_tags: np.ndarray,    # [m]
        active_episode: int,
    ) -> np.ndarray:                         # [r]  indices of top-r clusters
        self._step += 1
        if self._last_access is None:
            self._last_access = np.zeros(len(centroids), dtype=np.int64)

        scores = compute_retrieval_scores(
            query, centroids, cluster_episode_tags, active_episode,
            last_access=self._last_access,
            current_step=self._step,
            rho_eer=self.rho_eer,
            eta=self.eta,
            delta_rec=self.delta_rec,
        )

        top_r = np.argsort(-scores)[:self.r]

        # update last-access timestamps
        self._last_access[top_r] = self._step
        return top_r
