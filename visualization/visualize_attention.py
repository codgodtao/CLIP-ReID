"""
CLIP-REID Attention Visualization Script

Based on Transformer-Explainability method (https://github.com/hila-chefer/Transformer-Explainability)
CVPR 2021: Transformer Interpretability Beyond Attention Visualization

This script provides visualization of attention maps for CLIP-REID models.
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model.make_model_clipreid import make_model
from visualization.attention_vis import AttentionVisualizer, overlay_heatmap, visualize_attention_grid


class AttentionVisualizationApp:
    """GUI application for attention visualization."""

    def __init__(self, model, config_file, device='cuda'):
        self.model = model
        self.model.eval()
        self.config_file = config_file
        self.device = device
        self.visualizer = AttentionVisualizer(model)
        self.current_image = None
        self.current_image_path = None

        # Create GUI
        self.root = tk.Tk()
        self.root.title("CLIP-REID Attention Visualization")
        self.root.geometry("1200x800")

        self._create_widgets()

    def _create_widgets(self):
        """Create GUI widgets."""
        # Top frame for controls
        control_frame = tk.Frame(self.root, height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Load image button
        self.load_btn = tk.Button(control_frame, text="Load Image", command=self.load_image, width=15)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Visualize button
        self.vis_btn = tk.Button(control_frame, text="Generate Visualization", command=self.generate_visualization, width=20)
        self.vis_btn.pack(side=tk.LEFT, padx=5)
        self.vis_btn.config(state=tk.DISABLED)

        # Layer visualization button
        self.layer_btn = tk.Button(control_frame, text="Show All Layers", command=self.show_all_layers, width=15)
        self.layer_btn.pack(side=tk.LEFT, padx=5)
        self.layer_btn.config(state=tk.DISABLED)

        # Save button
        self.save_btn = tk.Button(control_frame, text="Save Result", command=self.save_result, width=15)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.config(state=tk.DISABLED)

        # Clear button
        self.clear_btn = tk.Button(control_frame, text="Clear", command=self.clear, width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Main content area
        content_frame = tk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Original image
        left_frame = tk.LabelFrame(content_frame, text="Original Image", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.original_ax = plt.figure(figsize=(5, 5))
        self.original_canvas = FigureCanvasTkAgg(self.original_ax, left_frame)
        self.original_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Middle panel - Attention heatmap
        middle_frame = tk.LabelFrame(content_frame, text="Attention Heatmap", padx=10, pady=10)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.heatmap_ax = plt.figure(figsize=(5, 5))
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_ax, middle_frame)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right panel - Overlay
        right_frame = tk.LabelFrame(content_frame, text="Overlay", padx=10, pady=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.overlay_ax = plt.figure(figsize=(5, 5))
        self.overlay_canvas = FigureCanvasTkAgg(self.overlay_ax, right_frame)
        self.overlay_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load an image to begin.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        """Load image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.current_image_path = file_path
                self.current_image = Image.open(file_path).convert('RGB')

                # Display original image
                self.original_ax.clear()
                ax = self.original_ax.add_subplot(111)
                ax.imshow(self.current_image)
                ax.axis('off')
                self.original_canvas.draw()

                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                self.vis_btn.config(state=tk.NORMAL)
                self.layer_btn.config(state=tk.DISABLED)
                self.save_btn.config(state=tk.DISABLED)

                # Clear other panels
                self.heatmap_ax.clear()
                self.heatmap_canvas.draw()
                self.overlay_ax.clear()
                self.overlay_canvas.draw()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess image for model input."""
        # Get image transforms from config
        size = cfg.INPUT.SIZE_TEST if hasattr(cfg, 'INPUT') else [256, 128]
        pixel_mean = cfg.INPUT.PIXEL_MEAN if hasattr(cfg, 'INPUT') else [0.485, 0.456, 0.406]
        pixel_std = cfg.INPUT.PIXEL_STD if hasattr(cfg, 'INPUT') else [0.229, 0.224, 0.225]

        # Resize
        image_resized = image.resize((size[1], size[0]), Image.BILINEAR)

        # Convert to tensor
        img_np = np.array(image_resized).astype(np.float32) / 255.0

        # Normalize
        img_np = (img_np - pixel_mean) / pixel_std

        # HWC to CHW
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, np.array(image)

    def generate_visualization(self):
        """Generate attention visualization for loaded image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            self.status_var.set("Processing...")

            # Preprocess image
            img_tensor, original_np = self.preprocess_image(self.current_image)

            # Generate visualization
            heatmap, attr_map, pred_class = self.visualizer.visualize_attention(
                img_tensor, original_np
            )

            # Display heatmap
            if heatmap is not None:
                self.heatmap_ax.clear()
                ax = self.heatmap_ax.add_subplot(111)
                ax.imshow(heatmap)
                ax.axis('off')
                self.heatmap_canvas.draw()

                # Display overlay
                overlay = overlay_heatmap(heatmap, original_np, alpha=0.5)
                self.overlay_ax.clear()
                ax = self.overlay_ax.add_subplot(111)
                ax.imshow(overlay)
                ax.axis('off')
                self.overlay_canvas.draw()

                self.save_btn.config(state=tk.NORMAL)

            self.status_var.set(f"Visualization generated. Predicted class: {pred_class.item()}")
            self.layer_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")
            self.status_var.set("Error occurred")

    def show_all_layers(self):
        """Show attention from all layers."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        try:
            self.status_var.set("Processing all layers...")

            # Preprocess image
            img_tensor, original_np = self.preprocess_image(self.current_image)

            # Generate layer visualization
            fig = visualize_attention_grid(img_tensor, self.current_image, self.model)

            # Show in new window
            fig_window = tk.Toplevel(self.root)
            fig_window.title("Attention Maps - All Layers")
            fig_window.geometry("1000x1000")

            canvas = FigureCanvasTkAgg(fig, fig_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.status_var.set("All layers visualization complete")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate layer visualization: {str(e)}")

    def save_result(self):
        """Save visualization results."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Nothing to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Visualization",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"attention_{os.path.basename(self.current_image_path)}"
        )

        if file_path:
            try:
                # Create figure with all three panels
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original
                axes[0].imshow(self.current_image)
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                # Get heatmap from canvas
                self.heatmap_ax.canvas.draw()
                heatmap_img = np.frombuffer(self.heatmap_ax.canvas.tostring_rgb(), dtype=np.uint8)
                heatmap_img = heatmap_img.reshape(self.heatmap_ax.canvas.get_width_height()[::-1] + (3,))

                axes[1].imshow(heatmap_img)
                axes[1].set_title("Attention Heatmap")
                axes[1].axis('off')

                # Get overlay from canvas
                self.overlay_ax.canvas.draw()
                overlay_img = np.frombuffer(self.overlay_ax.canvas.tostring_rgb(), dtype=np.uint8)
                overlay_img = overlay_img.reshape(self.overlay_ax.canvas.get_width_height()[::-1] + (3,))

                axes[2].imshow(overlay_img)
                axes[2].set_title("Overlay")
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                plt.close()

                self.status_var.set(f"Saved to: {file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def clear(self):
        """Clear all visualizations."""
        self.current_image = None
        self.current_image_path = None

        # Clear all panels
        self.original_ax.clear()
        self.original_canvas.draw()
        self.heatmap_ax.clear()
        self.heatmap_canvas.draw()
        self.overlay_ax.clear()
        self.overlay_canvas.draw()

        # Reset buttons
        self.vis_btn.config(state=tk.DISABLED)
        self.layer_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

        self.status_var.set("Ready. Load an image to begin.")

    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="CLIP-REID Attention Visualization")
    parser.add_argument("--config_file", required=True, help="path to config file")
    parser.add_argument("--model_weight", required=True, help="path to model weights")
    parser.add_argument("--device", default="cuda", help="device to use (cuda or cpu)")
    args = parser.parse_args()

    # Load config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Create output directory
    output_dir = cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create model
    print("Loading model...")
    model = make_model(cfg, num_class=cfg.DATASETS.NUM_CLASS, camera_num=0, view_num=0)
    model.load_param(args.model_weight)
    model.to(device)
    model.eval()

    print("Starting visualization app...")
    app = AttentionVisualizationApp(model, args.config_file, device)
    app.run()


if __name__ == "__main__":
    main()
