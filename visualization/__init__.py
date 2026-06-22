"""
CLIP-REID Attention Visualization Module

Based on Transformer-Explainability method (https://github.com/hila-chefer/Transformer-Explainability)
CVPR 2021: Transformer Interpretability Beyond Attention Visualization
"""

from .attention_vis import (
    AttentionVisualizer,
    overlay_heatmap,
    visualize_attention_grid
)

__all__ = [
    'AttentionVisualizer',
    'overlay_heatmap',
    'visualize_attention_grid'
]
