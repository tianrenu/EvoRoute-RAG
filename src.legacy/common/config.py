"""RAGarden 全局配置常量。"""
import os
from dataclasses import dataclass


@dataclass
class RAGardenConfig:
    """RAGarden 全局配置。"""

    # Qdrant 配置
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ragarden_campus_default"

    # MiniMax LLM 配置
    minimax_api_key: str = os.getenv("MINIMAX_API_KEY", "")
    minimax_base_url: str = "https://api.minimaxi.com/v1"
    minimax_model: str = "MiniMax-M2.7"

    # L2 Quality Gate
    quality_gate_threshold: float = 0.60  # 新Q公式整体偏低，0.60适配中文2-3字词覆盖率
    max_reretrieve_count: int = 3

    # L2 Generator
    generator_temperature: float = 0.3
    router_temperature: float = 0.1

    # Qdrant 检索参数
    retrieval_top_k: int = 3
    vector_dim: int = 512

    # L3 冲突检测
    conflict_block_threshold: float = 0.8
    conflict_warn_threshold: float = 0.65

    # 技能库
    skill_dir: str = "skills"
    skill_inflation_threshold: float = 0.30
    skill_dormant_days: int = 30

    # Q 公式权重（Quality Gate）
    q_weight_s: float = 0.40  # 相关性
    q_weight_m: float = 0.25  # 多样性
    q_weight_a: float = 0.20  # 答案支撑性
    q_weight_c: float = 0.15  # 完整性


# 全局单例（Phase 1 简化）
_config: RAGardenConfig | None = None


def get_config() -> RAGardenConfig:
    """获取全局配置单例。"""
    global _config
    if _config is None:
        _config = RAGardenConfig()
    return _config
