"""
CLIP-REID Attention Visualization - Command Line Interface

Based on Transformer-Explainability method (https://github.com/hila-chefer/Transformer-Explainability)
CVPR 2021: Transformer Interpretability Beyond Attention Visualization

Usage:
    python visualization/visualize_attention_cli.py --config_file configs/person/vit_clipreid.yml --model_weight path/to/weight.pth --image path/to/image.jpg

    # Show all layers
    python visualization/visualize_attention_cli.py --config_file configs/person/vit_clipreid.yml --model_weight path/to/weight.pth --image path/to/image.jpg --show_all_layers
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model.make_model_clipreid import make_model
from visualization.attention_vis import AttentionVisualizer, overlay_heatmap, visualize_attention_grid


def preprocess_image(image, size_test, pixel_mean, pixel_std):
    """Preprocess image for model input."""
    # Resize
    image_resized = image.resize((size_test[1], size_test[0]), Image.BILINEAR)

    # Convert to tensor
    img_np = np.array(image_resized).astype(np.float32) / 255.0

    # Normalize
    img_np = (img_np - pixel_mean) / pixel_std

    # HWC to CHW
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    return img_tensor, np.array(image)


def visualize_image(model, image_path, output_dir='output', config_file=None):
    """Visualize attention for a single image."""
    print(f"Processing: {image_path}")

    # Load image
    original_image = Image.open(image_path).convert('RGB')
    original_np = np.array(original_image)

    # Preprocess
    size_test = cfg.INPUT.SIZE_TEST if hasattr(cfg, 'INPUT') else [256, 128]
    pixel_mean = cfg.INPUT.PIXEL_MEAN if hasattr(cfg, 'INPUT') else [0.485, 0.456, 0.406]
    pixel_std = cfg.INPUT.PIXEL_STD if hasattr(cfg, 'INPUT') else [0.229, 0.224, 0.225]

    img_tensor, _ = preprocess_image(original_image, size_test, pixel_mean, pixel_std)

    # Create visualizer
    visualizer = AttentionVisualizer(model)

    # Generate visualization
    print("Generating attention visualization...")
    heatmap, attr_map, pred_class = visualizer.visualize_attention(img_tensor, original_np)

    # Create output
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Heatmap
    if heatmap is not None:
        axes[1].imshow(heatmap)
        axes[1].set_title(f"Attention Heatmap (Pred: {pred_class.item()})")
    else:
        axes[1].text(0.5, 0.5, "No attention data", ha='center', va='center')
        axes[1].set_title("Attention Heatmap")
    axes[1].axis('off')

    # Overlay
    if heatmap is not None:
        overlay = overlay_heatmap(heatmap, original_np, alpha=0.5)
        axes[2].imshow(overlay)
    else:
        axes[2].imshow(original_image)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_attention.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    # Also save attention map separately
    if attr_map is not None:
        attr_path = os.path.join(output_dir, f"{base_name}_attr_map.png")
        plt.imsave(attr_path, attr_map, cmap='jet')
        print(f"Saved attention map to: {attr_path}")

    plt.close()

    return output_path


def visualize_all_layers(model, image_path, output_dir='output'):
    """Visualize attention from all layers."""
    print(f"Processing: {image_path}")

    original_image = Image.open(image_path).convert('RGB')

    size_test = cfg.INPUT.SIZE_TEST if hasattr(cfg, 'INPUT') else [256, 128]
    pixel_mean = cfg.INPUT.PIXEL_MEAN if hasattr(cfg, 'INPUT') else [0.485, 0.456, 0.406]
    pixel_std = cfg.INPUT.PIXEL_STD if hasattr(cfg, 'INPUT') else [0.229, 0.224, 0.225]

    img_tensor, _ = preprocess_image(original_image, size_test, pixel_mean, pixel_std)

    print("Generating layer-wise attention visualization...")
    fig = visualize_attention_grid(img_tensor, original_image, model)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_all_layers.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="CLIP-REID Attention Visualization")
    parser.add_argument("--config_file", required=True, help="path to config file")
    parser.add_argument("--model_weight", required=True, help="path to model weights")
    parser.add_argument("--image", required=True, help="path to input image")
    parser.add_argument("--output_dir", default="output", help="output directory")
    parser.add_argument("--show_all_layers", action="store_true", help="show attention from all layers")
    args = parser.parse_args()

    # Load config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Loading model...")
    model = make_model(cfg, num_class=cfg.DATASETS.NUM_CLASS, camera_num=0, view_num=0)
    model.load_param(args.model_weight)
    model.to(device)
    model.eval()

    # Visualize
    if args.show_all_layers:
        visualize_all_layers(model, args.image, args.output_dir)
    else:
        visualize_image(model, args.image, args.output_dir, args.config_file)

    print("Done!")


if __name__ == "__main__":
    main()