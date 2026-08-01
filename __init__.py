# NarIndex Core — 三个机制的完整实现
#
# NarIndex: Narrative-Time-Aware Indexing for Coherent Long-Context LLM Inference
# PVLDB Vol. 20, 2027
#
# 用法:
#   python -m narindex.demo    ← 交互式演示（生成合成数据并展示三个机制）
#   python -m narindex.test    ← 快速验证 (同 pip install -e . 之后)

from .episode_field import (
    EpisodeField,
    temporal_phase_embedding,
    build_episode_field,
    spherical_kmeans_cluster,
)

from .conflict_potential import (
    ConflictDetector,
    compute_centroid_drift,
    compute_attention_divergence,
    semantic_conflict_potential,
)

from .eer import (
    EpisodeExclusiveRetrieval,
    compute_retrieval_scores,
    episode_softmax_bias,
)

from .narindex_model import NarIndex, NarIndexConfig

__all__ = [
    "EpisodeField",
    "temporal_phase_embedding",
    "build_episode_field",
    "spherical_kmeans_cluster",
    "ConflictDetector",
    "compute_centroid_drift",
    "compute_attention_divergence",
    "semantic_conflict_potential",
    "EpisodeExclusiveRetrieval",
    "compute_retrieval_scores",
    "episode_softmax_bias",
    "NarIndex",
    "NarIndexConfig",
]
