"""
CLIP-REID Attention Visualization Module

Based on Transformer-Explainability method (https://github.com/hila-chefer/Transformer-Explainability)
CVPR 2021: Transformer Interpretability Beyond Attention Visualization
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict


class GradHook:
    """Hook to capture gradients"""
    def __init__(self):
        self.gradients = None

    def __call__(self, grad):
        self.gradients = grad


class AttentionVisualizer:
    """
    Attention visualization based on Transformer-Explainability method.
    Implements LRP (Layer-wise Relevance Propagation) with attention rollout.
    """
    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.activations = []
        self.gradients = []
        self.hooks = []
        self.tokens = None

    def clear_cache(self):
        self.attentions = []
        self.activations = []
        self.gradients = []
        self.tokens = None

    def register_hooks(self):
        """Register forward and backward hooks on attention layers"""
        self.clear_cache()

        def get_attention(module, input, output):
            # Get attention weights from MultiheadAttention
            if hasattr(module, 'attn'):
                attn_output, attn_weights = module.attn(output[0], output[0], output[0], need_weights=True)
                self.attentions.append(attn_weights.detach())

        def get_activation(module, input, output):
            self.activations.append(output.detach())

        def get_gradient(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        # Register hooks on transformer blocks
        if hasattr(self.model.image_encoder, 'transformer'):
            for block in self.model.image_encoder.transformer.resblocks:
                self.hooks.append(block.attn.register_forward_hook(get_attention))
                self.hooks.append(block.attn.register_full_backward_hook(lambda module, grad_in, grad_out: self.gradients.append(grad_out[0].detach())))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_rollout(self, attentions, start_layer=0):
        """
        Compute attention rollout as described in Transformer-Explainability paper.
        Combines attention matrices across layers using matrix multiplication.
        """
        # Initialize with identity matrix
        num_tokens = attentions[0].shape[-1]
        rollout = torch.eye(num_tokens, num_tokens, device=attentions[0].device)

        # Add identity to attention weights (residual connection)
        for attention in attentions[start_layer:]:
            # attention shape: [batch, heads, tokens, tokens]
            if attention.ndim == 4:
                attention = attention.mean(dim=1)  # Average over heads
            # Add residual connection
            attention = attention + torch.eye(attention.shape[-1], device=attention.device)
            # Normalize
            attention = attention / attention.sum(dim=-1, keepdim=True)
            # Matrix multiply
            rollout = torch.matmul(attention, rollout)

        return rollout

    def compute_lrp(self, attentions, activations, gradients):
        """
        Compute LRP (Layer-wise Relevance Propagation) as described in the paper.
        Uses the Deep Taylor Decomposition principle.
        """
        if len(attentions) == 0 or len(gradients) == 0:
            return None

        # Use gradients to weight attention heads
        # Gradient represents importance of each head for the target class
        relevances = []

        for i, (attn, grad) in enumerate(zip(attentions, gradients)):
            if attn is None or grad is None:
                continue
            # attn: [batch, heads, tokens, tokens]
            # grad: [batch, tokens, heads]
            if grad.ndim == 3:
                grad = grad.permute(0, 2, 1)  # [batch, heads, tokens]
            # Compute head importance as gradient magnitude
            head_importance = grad.abs().mean(dim=-1, keepdim=True)  # [batch, heads, 1]
            # Weight attention by gradient
            weighted_attn = attn * head_importance
            relevances.append(weighted_attn)

        return relevances

    def compute_transformer_attribution(self, attentions, gradients):
        """
        Compute transformer attribution combining LRP and rollout.
        Based on Transformer-Explainability paper methodology.
        """
        if len(attentions) == 0:
            return None

        # Get attention from last layer
        attn = attentions[-1]
        if attn.ndim == 4:
            attn = attn.mean(dim=1)  # Average over heads

        # Use rollout to aggregate across layers
        rollout = self.compute_rollout(attentions)

        # If gradients available, use them to weight the result
        if len(gradients) > 0:
            grad = gradients[-1]
            if grad.ndim == 3:
                grad = grad.mean(dim=-1)  # Average over heads
                grad = grad.mean(dim=-1, keepdim=True)  # [batch, 1]
            # Weight the rollout by gradient
            rollout = rollout * grad

        return rollout

    def generate_heatmap(self, attribution, original_image, method='mix'):
        """
        Generate heatmap visualization from attribution scores.
        """
        if attribution is None:
            return None

        # Get the [CLS] token attribution (first token)
        if attribution.ndim == 3:
            cls_attr = attribution[0, 0, 1:].cpu().numpy()  # Skip CLS and SEP
        else:
            cls_attr = attribution[0, 1:].cpu().numpy()

        # Reshape to 2D grid (assuming square grid)
        seq_len = len(cls_attr)
        grid_size = int(np.sqrt(seq_len))

        if grid_size * grid_size == seq_len:
            attr_map = cls_attr.reshape(grid_size, grid_size)
        else:
            # Handle non-square case
            attr_map = cls_attr.reshape(1, -1)
            attr_map = cv2.resize(attr_map, (16, 16))

        # Normalize to 0-1
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        # Resize to original image size
        attr_map = cv2.resize(attr_map, (original_image.shape[1], original_image.shape[0]))

        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return heatmap, attr_map

    def visualize_attention(self, image_tensor, original_image, target_class=None):
        """
        Main method to generate attention visualization.
        """
        self.register_hooks()

        # Forward pass
        image_features_last, image_features, image_features_proj = self.model.image_encoder(image_tensor)

        # Get prediction logits
        if hasattr(self.model, 'classifier'):
            feat = self.model.bottleneck(image_features_last[:, 0])
            logits = self.model.classifier(feat)
        else:
            logits = image_features_last[:, 0]

        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=-1)

        # Backward pass
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Compute visualization
        if len(self.attentions) > 0:
            # Use transformer attribution method
            attribution = self.compute_transformer_attribution(self.attentions, self.gradients)
            heatmap, attr_map = self.generate_heatmap(attribution, original_image)
        else:
            # Fallback: use raw attention
            if len(self.activations) > 0:
                act = self.activations[-1]
                if act.ndim == 3:
                    attr = act[0, 0, 1:].cpu().numpy()
                    heatmap, attr_map = self.generate_heatmap(
                        attr.reshape(1, 1, -1) if attr.ndim == 1 else attr.unsqueeze(0),
                        original_image
                    )
                else:
                    heatmap = None
                    attr_map = None
            else:
                heatmap = None
                attr_map = None

        self.remove_hooks()

        return heatmap, attr_map, target_class


def overlay_heatmap(heatmap, original_image, alpha=0.5):
    """Overlay heatmap on original image."""
    if heatmap is None:
        return original_image

    # Resize heatmap to match image
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Normalize original image
    img_float = np.float32(original_image) / 255.0

    # Overlay
    overlay = (1 - alpha) * img_float + alpha * (np.float32(heatmap) / 255.0)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


def visualize_attention_grid(image_tensor, original_image, model, target_class=None):
    """
    Visualize attention from multiple layers.
    """
    visualizer = AttentionVisualizer(model)
    visualizer.register_hooks()

    # Forward pass
    image_features_last, image_features, image_features_proj = model.image_encoder(image_tensor)

    # Get prediction
    if hasattr(model, 'classifier'):
        feat = model.bottleneck(image_features_last[:, 0])
        logits = model.classifier(feat)
    else:
        logits = image_features_last[:, 0]

    if target_class is None:
        target_class = logits.argmax(dim=-1)

    # Backward pass
    model.zero_grad()
    logits[0, target_class].backward()

    # Generate visualizations for multiple layers
    num_layers = len(visualizer.attentions)
    grid_size = int(np.ceil(np.sqrt(num_layers)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten() if num_layers > 1 else [axes]

    original_img = np.array(original_image)

    for idx in range(num_layers):
        attn = visualizer.attentions[idx]
        if attn is not None and attn.ndim == 4:
            attn_mean = attn[0].mean(dim=0).cpu().numpy()
            # Get attention to [CLS] token
            cls_attn = attn_mean[:, 0]

            seq_len = len(cls_attn) - 1  # Exclude CLS
            grid = int(np.sqrt(seq_len))

            if grid * grid == seq_len:
                attn_map = cls_attn[1:].reshape(grid, grid)
            else:
                attn_map = cls_attn[1:].reshape(1, -1)
                attn_map = cv2.resize(attn_map, (16, 16))

            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_map = cv2.resize(attn_map, (original_img.shape[1], original_img.shape[0]))

            axes[idx].imshow(original_img)
            axes[idx].imshow(attn_map, cmap='jet', alpha=0.5)
            axes[idx].set_title(f'Layer {idx}')
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    visualizer.remove_hooks()

    return fig