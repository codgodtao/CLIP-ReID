"""CLIP-ReID Attention 可视化模块 - 基于 transformer-explanability 原理。

核心思路:
1. Hook transformer 的 attention 层，提取注意力权重
2. 使用 Gradient × Input 方法计算重要性分数
3. 将注意力权重映射到图像区域生成热力图

支持两种可视化方式:
1. Attention Map: 直接可视化注意力权重
2. Gradient Attention: 使用梯度加权的注意力图
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from typing import Dict, List, Tuple, Optional


class AttentionHook:
    """Hook transformer attention layers to extract attention weights."""

    def __init__(self, model: torch.nn.Module, layer_ids: Optional[List[int]] = None):
        self.model = model
        self.layer_ids = layer_ids
        self.attention_weights = {}
        self.hooks = []
        self._register_hooks()

    def _attention_hook(self, module, input, output):
        """Hook for attention layer output."""
        # output is (attn_output, attn_weights) for MultiheadAttention
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights = output[1]  # [heads, batch, seq_len, seq_len]
            layer_name = str(module.__class__.__name__) + f"_{len(self.attention_weights)}"
            self.attention_weights[layer_name] = attn_weights.detach().cpu()

    def _register_hooks(self):
        """Register hooks on all MultiheadAttention layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)

    def get_attention(self) -> Dict[str, torch.Tensor]:
        """Get collected attention weights."""
        return self.attention_weights

    def clear(self):
        """Clear collected attention weights."""
        self.attention_weights = {}

    def remove(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()


class AttentionVisualizer:
    """Visualize attention weights on images."""

    def __init__(self, image_size: Tuple[int, int] = (256, 128), patch_size: int = 16):
        self.image_size = image_size  # (H, W)
        self.patch_size = patch_size
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.total_patches = self.num_patches_h * self.num_patches_w + 1  # +1 for CLS token

        # Custom colormap (blue to red)
        self.cmap = LinearSegmentedColormap.from_list(
            'attention',
            [(0.0, '#1e3a5f'), (0.5, '#3498db'), (1.0, '#e74c3c')]
        )

    def _normalize_attention(self, attn_weights: np.ndarray) -> np.ndarray:
        """Normalize attention weights to [0, 1]."""
        min_val = np.min(attn_weights)
        max_val = np.max(attn_weights)
        if max_val - min_val < 1e-6:
            return np.zeros_like(attn_weights)
        return (attn_weights - min_val) / (max_val - min_val)

    def _create_heatmap(self, attention: np.ndarray, method: str = 'mean') -> np.ndarray:
        """Create attention heatmap from attention weights."""
        # attention shape: [num_heads, seq_len, seq_len]
        
        # Focus on CLS token attention (first token attends to all patches)
        cls_attention = attention[:, 0, 1:]  # [heads, num_patches]
        
        if method == 'mean':
            heatmap = np.mean(cls_attention, axis=0)
        elif method == 'max':
            heatmap = np.max(cls_attention, axis=0)
        elif method == 'sum':
            heatmap = np.sum(cls_attention, axis=0)
        else:
            heatmap = np.mean(cls_attention, axis=0)
        
        # Reshape to 2D
        heatmap = heatmap.reshape(self.num_patches_h, self.num_patches_w)
        return heatmap

    def _upsample_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Upsample heatmap to match image size."""
        return cv2.resize(heatmap, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_CUBIC)

    def visualize(self, image: Image.Image, attention: np.ndarray, method: str = 'mean') -> Image.Image:
        """
        Generate attention visualization overlay on image.
        
        Args:
            image: Input PIL Image
            attention: Attention weights array [num_heads, seq_len, seq_len]
            method: Aggregation method ('mean', 'max', 'sum')
        
        Returns:
            PIL Image with attention heatmap overlay
        """
        # Convert image to numpy
        img_np = np.array(image.resize((self.image_size[1], self.image_size[0]))).astype(np.float32) / 255.0

        # Create heatmap
        heatmap = self._create_heatmap(attention, method)
        heatmap = self._normalize_attention(heatmap)
        heatmap = self._upsample_heatmap(heatmap)

        # Apply colormap
        heatmap_colored = self.cmap(heatmap)[:, :, :3]  # RGB only

        # Overlay: 70% image + 30% heatmap
        overlay = 0.7 * img_np + 0.3 * heatmap_colored
        
        # Convert back to PIL
        overlay = (overlay * 255).astype(np.uint8)
        return Image.fromarray(overlay)

    def visualize_layer_attention(self, image: Image.Image, attention_dict: Dict[str, np.ndarray]) -> List[Image.Image]:
        """Visualize attention from multiple layers."""
        results = []
        for layer_name, attn_weights in attention_dict.items():
            vis = self.visualize(image, attn_weights[0].numpy())  # First batch
            vis.info = {"layer": layer_name}
            results.append(vis)
        return results


class GradientAttention:
    """Gradient-based attention visualization (Gradient × Input method)."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

    def _gradient_hook(self, module, grad_input, grad_output):
        """Hook to capture gradients."""
        self.gradients = grad_output[0].detach().cpu()

    def _activation_hook(self, module, input, output):
        """Hook to capture activations."""
        self.activations = output.detach().cpu()

    def compute_attention(self, image_tensor: torch.Tensor, target_layer: torch.nn.Module):
        """
        Compute gradient-based attention.
        
        Args:
            image_tensor: [1, 3, H, W] input tensor
            target_layer: Layer to compute attention for
        
        Returns:
            Attention map as numpy array
        """
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks
        grad_hook = target_layer.register_backward_hook(self._gradient_hook)
        act_hook = target_layer.register_forward_hook(self._activation_hook)

        # Forward pass
        image_tensor.requires_grad = True
        output = self.model(image_tensor)
        
        # Backward pass on first feature (CLS token)
        if isinstance(output, tuple):
            output = output[0] if isinstance(output[0], torch.Tensor) else output[0][0]
        
        # Get the CLS token feature and backprop
        cls_feature = output[:, 0, :] if output.dim() == 3 else output[:, 0]
        cls_feature.sum().backward()

        # Remove hooks
        grad_hook.remove()
        act_hook.remove()

        # Compute Gradient × Activation
        if self.gradients is not None and self.activations is not None:
            weights = self.gradients.mean(dim=(1, 2), keepdim=True)
            attention = (weights * self.activations).sum(dim=1)
            return attention.numpy()[0]
        
        return None


def generate_attention_visualization(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: Tuple[int, int] = (256, 128),
    patch_size: int = 16,
    method: str = 'mean'
) -> Dict[str, Image.Image]:
    """
    Generate attention visualization for CLIP-ReID model.
    
    Args:
        model: CLIP-ReID model (build_transformer output)
        image: Input PIL Image
        image_size: Target image size (H, W)
        patch_size: ViT patch size
        method: Attention aggregation method
    
    Returns:
        Dictionary of visualization results
    """
    # Preprocess image
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(next(model.parameters()).device)

    # Get image encoder
    if hasattr(model, 'image_encoder'):
        encoder = model.image_encoder
    elif hasattr(model, 'module') and hasattr(model.module, 'image_encoder'):
        encoder = model.module.image_encoder
    else:
        raise ValueError("Cannot find image encoder in model")

    # Register hooks
    hook = AttentionHook(encoder)

    # Forward pass
    model.eval()
    with torch.no_grad():
        encoder(image_tensor)

    # Get attention weights
    attention_dict = hook.get_attention()
    hook.remove()

    # Generate visualizations
    visualizer = AttentionVisualizer(image_size, patch_size)
    results = {}

    for layer_name, attn_weights in attention_dict.items():
        vis = visualizer.visualize(image, attn_weights[0].numpy(), method)
        results[layer_name] = vis

    # Also generate overall attention map
    all_attention = np.concatenate([v[0].numpy() for v in attention_dict.values()], axis=0)
    avg_attention = np.mean(all_attention, axis=0)
    results['overall'] = visualizer.visualize(image, avg_attention, method)

    return results


def save_visualization(results: Dict[str, Image.Image], output_dir: str = 'attention_viz'):
    """Save visualization results to directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, image in results.items():
        image.save(os.path.join(output_dir, f'attention_{layer_name}.png'))
        print(f"Saved: {layer_name}.png")


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.insert(0, '/workspace')
    
    from model.make_model_mobileclip2 import build_transformer
    from config import cfg
    
    # Load model
    cfg.merge_from_file('configs/person/vit_mobileclip2.yml')
    cfg.freeze()
    
    model = build_transformer(num_classes=1000, camera_num=1, view_num=1, cfg=cfg)
    
    # Load example image
    test_image = Image.open('test.jpg').convert('RGB')
    
    # Generate visualization
    results = generate_attention_visualization(model, test_image)
    
    # Save results
    save_visualization(results)
