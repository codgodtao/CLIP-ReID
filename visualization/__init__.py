"""
CLIP-REID Attention Visualization Module

提供多种可视化方法：
1. ViTAttentionExtractor - 从 Vision Transformer 提取 attention weights
2. Attention Rollout - 基于 Abnar & Zuidema (2020) 的聚合方法
3. GradCAM-style visualization - 基于梯度的可视化

Reference:
- "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
- "Transformer Interpretability Beyond Attention Visualization" (Chefer et al., CVPR 2021)
"""

# 新的 ViT attention 可视化 (推荐使用)
from .vit_attention_vis import (
    ViTAttentionExtractor,
    compute_attention_rollout,
    visualize_vit_attention,
    visualize_all_layers,
    visualize_attention_rollout,
    GradCAMViT
)

# 旧的 attention_vis (保留向后兼容，但建议使用新的)
from .attention_vis import (
    AttentionVisualizer,
    overlay_heatmap,
    visualize_attention_grid
)

__all__ = [
    # 新的 ViT 可视化
    'ViTAttentionExtractor',
    'compute_attention_rollout',
    'visualize_vit_attention',
    'visualize_all_layers',
    'visualize_attention_rollout',
    'GradCAMViT',
    # 旧的 (向后兼容)
    'AttentionVisualizer',
    'overlay_heatmap',
    'visualize_attention_grid'
]