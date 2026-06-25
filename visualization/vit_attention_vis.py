"""
ViT Attention Visualization for CLIP-ReID

基于业界标准方法实现：
1. Attention Rollout (Abnar & Zuidema, 2020)
2. Raw Attention Maps
3. GradCAM-style visualization

Reference:
- "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
- "Transformer Interpretability Beyond Attention Visualization" (Chefer et al., CVPR 2021)
- CLIP official implementation
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model.make_model_clipreid import make_model


class ViTAttentionExtractor:
    """
    从 Vision Transformer 中提取 attention weights。

    支持 CLIP-ReID 中的 VisionTransformer 结构：
    - conv1: patch embedding
    - class_embedding: CLS token
    - positional_embedding
    - transformer.resblocks: 包含 MultiheadAttention
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attention_maps: List[torch.Tensor] = []
        self.hooks = []

    def clear(self):
        """清除缓存的 attention maps 和 hooks"""
        self.attention_maps = []
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def register_hooks(self):
        """
        注册 forward hooks 来捕获 attention weights。

        VisionTransformer 结构：
        - image_encoder (VisionTransformer)
          - transformer (Transformer)
            - resblocks (List[ResidualAttentionBlock])
              - attn (nn.MultiheadAttention)
        """
        self.clear()

        def get_attention_hook(module, input, output):
            """
            nn.MultiheadAttention 的 forward 返回：
            (attn_output, attn_weights) 当 need_weights=True
            或只返回 attn_output 当 need_weights=False

            我们需要手动计算 attention weights。
            """
            # input: (query, key, value)
            # 对于 self-attention: query = key = value = x
            query, key, value = input

            # 获取模块参数
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = embed_dim // num_heads

            # 计算 Q, K, V projections
            # module.in_proj_weight 包含 q,k,v 的权重 (concatenated)
            # module.in_proj_bias 包含 q,k,v 的 bias
            q_proj_weight = module.q_proj_weight if hasattr(module, 'q_proj_weight') else module.in_proj_weight[:embed_dim]
            k_proj_weight = module.k_proj_weight if hasattr(module, 'k_proj_weight') else module.in_proj_weight[embed_dim:2*embed_dim]
            v_proj_weight = module.v_proj_weight if hasattr(module, 'v_proj_weight') else module.in_proj_weight[2*embed_dim:]

            q_proj_bias = module.q_proj_bias if hasattr(module, 'q_proj_bias') else module.in_proj_bias[:embed_dim]
            k_proj_bias = module.k_proj_bias if hasattr(module, 'k_proj_bias') else module.in_proj_bias[embed_dim:2*embed_dim]
            v_proj_bias = module.v_proj_bias if hasattr(module, 'v_proj_bias') else module.in_proj_bias[2*embed_dim:]

            # 计算 Q, K, V
            q = F.linear(query, q_proj_weight, q_proj_bias)
            k = F.linear(key, k_proj_weight, k_proj_bias)
            v = F.linear(value, v_proj_weight, v_proj_bias)

            # Reshape for multi-head attention
            # (seq_len, batch, embed_dim) -> (batch, num_heads, seq_len, head_dim)
            batch_size = q.shape[1]
            seq_len = q.shape[0]

            q = q.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
            k = k.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
            v = v.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)

            # 计算 attention scores
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply softmax
            attn_weights = F.softmax(attn_weights, dim=-1)

            # 保存: (batch, num_heads, seq_len, seq_len)
            self.attention_maps.append(attn_weights.detach())

        # 注册 hooks 到所有 transformer blocks
        if hasattr(self.model, 'image_encoder'):
            image_encoder = self.model.image_encoder
            if hasattr(image_encoder, 'transformer'):
                transformer = image_encoder.transformer
                if hasattr(transformer, 'resblocks'):
                    for block in transformer.resblocks:
                        if hasattr(block, 'attn'):
                            hook = block.attn.register_forward_hook(get_attention_hook)
                            self.hooks.append(hook)

    def get_attention_maps(self) -> List[torch.Tensor]:
        """获取所有层的 attention maps"""
        return self.attention_maps

    def get_grid_size(self) -> Tuple[int, int]:
        """
        获取 patch grid 的尺寸。

        VisionTransformer 的 positional_embedding 形状为 [h*w + 1, width]
        其中 h*w 是 patch 数量，+1 是 CLS token。
        """
        if hasattr(self.model, 'image_encoder'):
            encoder = self.model.image_encoder
            if hasattr(encoder, 'h_resolution') and hasattr(encoder, 'w_resolution'):
                return encoder.h_resolution, encoder.w_resolution
            if hasattr(encoder, 'positional_embedding'):
                num_patches = encoder.positional_embedding.shape[0] - 1
                grid_size = int(np.sqrt(num_patches))
                return grid_size, grid_size
        return 16, 8  # 默认值 (256/16, 128/16)


def compute_attention_rollout(attention_maps: List[torch.Tensor],
                               start_layer: int = 0,
                               include_residual: bool = True) -> torch.Tensor:
    """
    计算 Attention Rollout (Abnar & Zuidema, 2020)

    通过矩阵乘法聚合所有层的 attention，得到每个 token 对最终输出的贡献。

    Args:
        attention_maps: 各层的 attention weights, 每个形状为 (batch, heads, seq, seq)
        start_layer: 开始聚合的层索引
        include_residual: 是否考虑残差连接 (添加 identity matrix)

    Returns:
        rollout: (batch, seq, seq) 的聚合 attention matrix
    """
    if len(attention_maps) == 0:
        return None

    # 获取第一层的信息
    batch_size = attention_maps[0].shape[0]
    num_tokens = attention_maps[0].shape[-1]
    device = attention_maps[0].device

    # 初始化 rollout 为 identity matrix
    rollout = torch.eye(num_tokens, num_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    # 逐层聚合
    for i in range(start_layer, len(attention_maps)):
        attn = attention_maps[i]  # (batch, heads, seq, seq)

        # 平均所有 heads
        attn = attn.mean(dim=1)  # (batch, seq, seq)

        # 添加残差连接 (identity)
        if include_residual:
            attn = attn + torch.eye(num_tokens, num_tokens, device=device).unsqueeze(0)
            # 归一化
            attn = attn / attn.sum(dim=-1, keepdim=True)

        # 矩阵乘法聚合
        rollout = torch.matmul(attn, rollout)

    return rollout


def get_cls_attention(rollout: torch.Tensor) -> torch.Tensor:
    """
    从 rollout matrix 中提取 CLS token 对其他 tokens 的 attention。

    Args:
        rollout: (batch, seq, seq)

    Returns:
        cls_attention: (batch, seq-1) CLS token 对各 patch 的 attention
    """
    # CLS token 是第一个 token (index 0)
    # rollout[0, i] 表示 token i 对最终输出的贡献
    # 我们取 rollout[:, 0, 1:] 表示 CLS token 对各 patch 的 attention
    cls_attention = rollout[:, 0, 1:]  # 跳过 CLS token 自身
    return cls_attention


def reshape_attention_to_2d(attn: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    将 1D attention reshape 为 2D grid。

    Args:
        attn: (batch, num_patches) 或 (num_patches)
        grid_h: grid 高度
        grid_w: grid 宽度

    Returns:
        attn_2d: (batch, grid_h, grid_w) 或 (grid_h, grid_w)
    """
    if attn.ndim == 1:
        return attn.reshape(grid_h, grid_w)
    elif attn.ndim == 2:
        return attn.reshape(attn.shape[0], grid_h, grid_w)
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")


def normalize_attention(attn: torch.Tensor) -> torch.Tensor:
    """归一化 attention 到 [0, 1]"""
    attn = attn - attn.min()
    attn = attn / (attn.max() + 1e-8)
    return attn


def create_heatmap(attn_2d: np.ndarray, image_size: Tuple[int, int],
                   colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    创建热力图可视化。

    Args:
        attn_2d: 2D attention map (已经归一化到 [0, 1])
        image_size: (height, width) 目标图像尺寸
        colormap: OpenCV colormap 类型

    Returns:
        heatmap: RGB 热力图
    """
    # Resize 到目标尺寸
    heatmap = cv2.resize(attn_2d, (image_size[1], image_size[0]))

    # 应用 colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


def overlay_heatmap_on_image(heatmap: np.ndarray, image: np.ndarray,
                              alpha: float = 0.5) -> np.ndarray:
    """
    将热力图叠加到原始图像上。

    Args:
        heatmap: RGB 热力图
        image: 原始 RGB 图像
        alpha: 热力图透明度

    Returns:
        overlay: 叠加后的图像
    """
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def preprocess_image(image: Image.Image, cfg) -> Tuple[torch.Tensor, np.ndarray]:
    """
    预处理图像用于模型输入。

    Args:
        image: PIL Image
        cfg: 配置对象

    Returns:
        img_tensor: 预处理后的 tensor (1, C, H, W)
        original_np: 原始图像 numpy array
    """
    size_test = cfg.INPUT.SIZE_TEST if hasattr(cfg.INPUT, 'SIZE_TEST') else [256, 128]
    pixel_mean = cfg.INPUT.PIXEL_MEAN if hasattr(cfg.INPUT, 'PIXEL_MEAN') else [0.5, 0.5, 0.5]
    pixel_std = cfg.INPUT.PIXEL_STD if hasattr(cfg.INPUT, 'PIXEL_STD') else [0.5, 0.5, 0.5]

    # Resize
    image_resized = image.resize((size_test[1], size_test[0]), Image.BILINEAR)

    # Convert to numpy and normalize
    img_np = np.array(image_resized).astype(np.float32) / 255.0
    img_np = (img_np - np.array(pixel_mean)) / np.array(pixel_std)

    # HWC to CHW
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)

    return img_tensor, np.array(image)


def visualize_single_layer_attention(attn: torch.Tensor, grid_h: int, grid_w: int,
                                      image: np.ndarray, layer_idx: int) -> np.ndarray:
    """
    可视化单层的 attention。

    Args:
        attn: (batch, heads, seq, seq) 或 (heads, seq, seq)
        grid_h, grid_w: patch grid 尺寸
        image: 原始图像
        layer_idx: 层索引

    Returns:
        heatmap: 热力图
    """
    if attn.ndim == 4:
        attn = attn[0]  # 取第一个 batch

    # 平均所有 heads
    attn = attn.mean(dim=0)  # (seq, seq)

    # 取 CLS token 对其他 tokens 的 attention
    cls_attn = attn[0, 1:]  # (seq-1)

    # Reshape to 2D
    cls_attn = cls_attn.reshape(grid_h, grid_w)

    # Normalize
    cls_attn = normalize_attention(cls_attn)

    # Create heatmap
    heatmap = create_heatmap(cls_attn.cpu().numpy(), image.shape[:2])

    return heatmap


def visualize_all_layers(attention_maps: List[torch.Tensor], grid_h: int, grid_w: int,
                          image: np.ndarray, output_path: str = None) -> plt.Figure:
    """
    可视化所有层的 attention。

    Args:
        attention_maps: 各层的 attention
        grid_h, grid_w: grid 尺寸
        image: 原始图像
        output_path: 保存路径 (可选)

    Returns:
        fig: matplotlib Figure
    """
    num_layers = len(attention_maps)
    grid_size = int(np.ceil(np.sqrt(num_layers + 1)))  # +1 for original image

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()

    # 第一张是原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    # 其余是各层 attention
    for i, attn in enumerate(attention_maps):
        heatmap = visualize_single_layer_attention(attn, grid_h, grid_w, image, i)
        axes[i + 1].imshow(heatmap)
        axes[i + 1].set_title(f'Layer {i}', fontsize=10)
        axes[i + 1].axis('off')

    # 隐藏多余的 axes
    for i in range(num_layers + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    return fig


def visualize_attention_rollout(attention_maps: List[torch.Tensor], grid_h: int, grid_w: int,
                                  image: np.ndarray, start_layer: int = 0) -> np.ndarray:
    """
    可视化 Attention Rollout 结果。

    Args:
        attention_maps: 各层的 attention
        grid_h, grid_w: grid 尺寸
        image: 原始图像
        start_layer: 开始聚合的层

    Returns:
        heatmap: 热力图
    """
    rollout = compute_attention_rollout(attention_maps, start_layer)
    if rollout is None:
        return None

    cls_attn = get_cls_attention(rollout)  # (batch, seq-1)
    cls_attn = cls_attn[0]  # 取第一个 batch

    # Reshape to 2D
    cls_attn = reshape_attention_to_2d(cls_attn, grid_h, grid_w)

    # Normalize
    cls_attn = normalize_attention(cls_attn)

    # Create heatmap
    heatmap = create_heatmap(cls_attn.cpu().numpy(), image.shape[:2])

    return heatmap


class GradCAMViT:
    """
    GradCAM-style visualization for ViT.

    基于梯度的可视化方法，计算特定输出对输入 patches 的梯度，
    得到每个 patch 对输出的重要性。
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None

    def register_hooks(self):
        """注册 hooks 捕获梯度和激活"""
        self.clear()

        def forward_hook(module, input, output):
            # 保存最后一层 transformer block 的输出
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # 保存梯度
            self.gradients = grad_output[0].detach()

        # 注册到最后一层 transformer block
        if hasattr(self.model, 'image_encoder'):
            encoder = self.model.image_encoder
            if hasattr(encoder, 'transformer'):
                transformer = encoder.transformer
                if hasattr(transformer, 'resblocks'):
                    last_block = transformer.resblocks[-1]
                    self.hooks.append(last_block.register_forward_hook(forward_hook))
                    self.hooks.append(last_block.register_full_backward_hook(backward_hook))

    def clear(self):
        self.gradients = None
        self.activations = None
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_cam(self, target_class: Optional[int] = None) -> torch.Tensor:
        """
        计算 CAM (Class Activation Map)。

        Args:
            target_class: 目标类别 (可选，默认使用预测类别)

        Returns:
            cam: 2D activation map
        """
        if self.gradients is None or self.activations is None:
            return None

        # gradients: (seq, batch, dim)
        # activations: (seq, batch, dim)

        # 计算每个 token 的权重 (梯度平均)
        weights = self.gradients.mean(dim=0, keepdim=True)  # (1, batch, dim)
        weights = weights.mean(dim=-1, keepdim=True)  # (1, batch, 1)

        # 计算 CAM: weights * activations
        cam = (weights * self.activations).sum(dim=-1)  # (seq, batch)
        cam = cam.permute(1, 0)  # (batch, seq)

        # 取 CLS token 和 patch tokens
        # CLS token 是第一个，我们关注它对 patches 的贡献
        cam = cam[:, 1:]  # 跳过 CLS token

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0]  # 返回第一个 batch


def visualize_vit_attention(model, image_path: str, cfg,
                             output_dir: str = 'output',
                             method: str = 'rollout',
                             show_all_layers: bool = False,
                             device: str = 'cuda'):
    """
    主可视化函数。

    Args:
        model: CLIP-ReID 模型
        image_path: 输入图像路径
        cfg: 配置对象
        output_dir: 输出目录
        method: 可视化方法 ('rollout', 'raw', 'gradcam', 'all')
        show_all_layers: 是否显示所有层
        device: 设备
    """
    # 加载图像
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image)

    # 预处理
    img_tensor, _ = preprocess_image(original_image, cfg)
    img_tensor = img_tensor.to(device)

    # 创建 attention extractor
    extractor = ViTAttentionExtractor(model, device)
    extractor.register_hooks()

    # Forward pass
    model.eval()
    with torch.no_grad():
        # 获取模型输出
        if hasattr(model, 'image_encoder'):
            x11, x12, xproj = model.image_encoder(img_tensor)

            # 获取预测
            if hasattr(model, 'bottleneck'):
                feat = model.bottleneck(x12[:, 0])
                if hasattr(model, 'classifier'):
                    logits = model.classifier(feat)
                    pred_class = logits.argmax(dim=-1).item()
                else:
                    pred_class = 0
            else:
                pred_class = 0
        else:
            pred_class = 0

    # 获取 attention maps
    attention_maps = extractor.get_attention_maps()
    grid_h, grid_w = extractor.get_grid_size()

    print(f"Number of attention layers: {len(attention_maps)}")
    print(f"Grid size: {grid_h} x {grid_w}")
    print(f"Predicted class: {pred_class}")

    if len(attention_maps) == 0:
        print("Warning: No attention maps captured!")
        extractor.clear()
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 可视化
    if method == 'all' or show_all_layers:
        # 显示所有层
        output_path = os.path.join(output_dir, f"{base_name}_all_layers.png")
        visualize_all_layers(attention_maps, grid_h, grid_w, original_np, output_path)

    if method == 'rollout' or method == 'all':
        # Attention Rollout
        heatmap = visualize_attention_rollout(attention_maps, grid_h, grid_w, original_np)
        if heatmap is not None:
            overlay = overlay_heatmap_on_image(heatmap, original_np, alpha=0.5)

            # 保存
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_np)
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(heatmap)
            axes[1].set_title('Attention Rollout')
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{base_name}_rollout.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved rollout to: {output_path}")
            plt.close()

    if method == 'raw' or method == 'all':
        # Raw attention (最后一层)
        last_attn = attention_maps[-1]
        heatmap = visualize_single_layer_attention(last_attn, grid_h, grid_w, original_np, len(attention_maps)-1)
        overlay = overlay_heatmap_on_image(heatmap, original_np, alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(heatmap)
        axes[1].set_title(f'Raw Attention (Layer {len(attention_maps)-1})')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{base_name}_raw.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved raw attention to: {output_path}")
        plt.close()

    if method == 'gradcam' or method == 'all':
        # GradCAM-style
        gradcam = GradCAMViT(model, device)
        gradcam.register_hooks()

        # Forward + Backward
        model.zero_grad()
        x11, x12, xproj = model.image_encoder(img_tensor)

        if hasattr(model, 'bottleneck') and hasattr(model, 'classifier'):
            feat = model.bottleneck(x12[:, 0])
            logits = model.classifier(feat)
            target_class = logits.argmax(dim=-1)
            logits[0, target_class].backward()
        else:
            xproj[0, 0].sum().backward()

        cam = gradcam.compute_cam()
        if cam is not None:
            cam_2d = reshape_attention_to_2d(cam, grid_h, grid_w)
            heatmap = create_heatmap(cam_2d.cpu().numpy(), original_np.shape[:2])
            overlay = overlay_heatmap_on_image(heatmap, original_np, alpha=0.5)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_np)
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(heatmap)
            axes[1].set_title('GradCAM')
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{base_name}_gradcam.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved GradCAM to: {output_path}")
            plt.close()

        gradcam.clear()

    # 清理
    extractor.clear()
    print("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description="ViT Attention Visualization for CLIP-ReID")
    parser.add_argument("--config_file", required=True, help="path to config file")
    parser.add_argument("--model_weight", required=True, help="path to model weights")
    parser.add_argument("--image", required=True, help="path to input image")
    parser.add_argument("--output_dir", default="output/attention_vis", help="output directory")
    parser.add_argument("--method", default="all",
                        choices=['rollout', 'raw', 'gradcam', 'all'],
                        help="visualization method")
    parser.add_argument("--show_all_layers", action="store_true",
                        help="show attention from all layers")
    parser.add_argument("--device", default="cuda", help="device to use")
    args = parser.parse_args()

    # Load config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Loading model...")
    num_classes = cfg.DATASETS.NUM_CLASS if hasattr(cfg.DATASETS, 'NUM_CLASS') else 1000
    model = make_model(cfg, num_class=num_classes, camera_num=0, view_num=0)
    model.load_param(args.model_weight)
    model.to(device)
    model.eval()
    print("Model loaded!")

    # Visualize
    visualize_vit_attention(
        model=model,
        image_path=args.image,
        cfg=cfg,
        output_dir=args.output_dir,
        method=args.method,
        show_all_layers=args.show_all_layers,
        device=device
    )


if __name__ == "__main__":
    main()