"""
NarIndex  —  完整的端到端模型
================================

整合三个机制为单个 NarIndex 推理模块，对应 Algorithm 1。
"""

from __future__ import annotations

__all__ = ["NarIndexConfig", "NarIndex"]

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .episode_field import EpisodeField, build_episode_field
from .conflict_potential import ConflictDetector
from .eer import EpisodeExclusiveRetrieval, episode_softmax_bias


# ---------------------------------------------------------------------------
#  配置
# ---------------------------------------------------------------------------

@dataclass
class NarIndexConfig:
    """NarIndex 三个机制的默认超参数 (from paper, §3)."""

    # Episode field
    d: int = 128           # key dim per head
    r: int = 8             # temporal-phase bands
    lam: float = 0.8       # λ: semantic-temporal trade-off
    theta: float = 1e6     # RoPE base Θ
    m: int = 8             # initial episode count

    # Conflict detector
    rho_ewma: float = 0.1   # ρ_EWMA
    kappa: float = 2.0      # κ
    gamma: float = 1.0      # γ  (attention-divergence weight)
    burn_in: int = 512       # T_burn
    refractory: int = 64     # T_refr

    # EER
    rho_eer: float = 1.0    # ρ_EER
    eta: float = 0.15       # recency-weight η
    delta_rec: float = 0.001
    r_retrieval: int = 64   # retrieval budget


# ---------------------------------------------------------------------------
#  NarIndex
# ---------------------------------------------------------------------------

@dataclass
class NarIndex:
    """
    端到端的 NarIndex 推理模型。

    使用:
      >>> model = NarIndex(NarIndexConfig(d=128))
      >>> model.build(keys, positions)               # 构造 episode-field index
      >>> o = model.decode_step(query, keys, values) # 执行一个解码步
    """

    config: NarIndexConfig = field(default_factory=NarIndexConfig)

    # sub-modules (lazy init)
    _ef: Optional[EpisodeField] = field(default=None)
    _detector: Optional[ConflictDetector] = field(default=None)
    _eer: Optional[EpisodeExclusiveRetrieval] = field(default=None)

    # state
    active_episode: int = 0
    step: int = 0

    def build(self, keys: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        构造 NarIndex episode-field 索引。
        返回 [n] cluster labels.
        """
        cfg = self.config
        self._ef = EpisodeField(d=cfg.d, r=cfg.r, lam=cfg.lam, theta=cfg.theta, m=cfg.m)
        labels = self._ef.build(keys, positions)
        self._detector = ConflictDetector(
            rho=cfg.rho_ewma, kappa=cfg.kappa, gamma=cfg.gamma,
            burn_in=cfg.burn_in, refractory=cfg.refractory,
        )
        self._eer = EpisodeExclusiveRetrieval(
            rho_eer=cfg.rho_eer, eta=cfg.eta, delta_rec=cfg.delta_rec,
            r=cfg.r_retrieval,
        )
        self.active_episode = 0
        self.step = 0
        return labels

    def decode_step(
        self,
        query: np.ndarray,    # [d]
        keys: np.ndarray,     # [n, d]    stored keys
        values: np.ndarray,   # [n, d]    stored values
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Algorithm 1 — 单步解码。

        Returns:
          output_vec:  ndarray [d]     加权 value 输出
          retrieval_zone: list[int]    被检索的 cluster indices
        """
        if self._ef is None or self._eer is None or self._detector is None:
            raise RuntimeError("Call .build() before decode_step()")

        centroids_k = self._ef.centroids_[:, : self.config.d]        # [m, d]
        cluster_sizes = np.array([
            (self._ef.labels_ == c).sum() for c in range(self.config.m)
        ])
        cluster_episode_tags = np.arange(self.config.m) % self.config.m

        # 1. EER: select top-r retrieval zone  (Eq. 5)
        retrieval_zone = self._eer.select_retrieval_zone(
            query, centroids_k, cluster_episode_tags, self.active_episode,
        )

        # 2. precise attention in retrieval zone  (Eq. 6)
        n = len(keys)
        retrieval_mask = np.isin(self._ef.labels_, retrieval_zone)
        if retrieval_mask.sum() == 0:
            output = np.zeros(self.config.d)
        else:
            attn_w = episode_softmax_bias(
                query, keys, self._ef.labels_, cluster_episode_tags,
                self.active_episode, rho_eer=self.config.rho_eer, d=self.config.d,
            )
            output = values.T @ attn_w

        # 3. update conflict potential  (Eq. 3–4)
        attn = np.zeros(self.config.m)
        attn[retrieval_zone] = 1.0 / max(len(retrieval_zone), 1)
        boundary = self._detector.step(centroids_k, cluster_sizes, attn)

        if boundary:
            self.active_episode += 1

        self.step += 1
        return output, retrieval_zone.tolist()
