"""训练结束后导出 ONNX — 支持 CLIP-ReID 和 MobileCLIP2-ReID 两种模型。

统一的 ONNX 接口：
    输入:  images  [N, 3, H, W]  float32, 已归一化
    输出:  features [N, D]        float32, L2 归一化 (可选)

deploy_config.json 包含预处理参数，场外推理只需 onnxruntime + numpy + Pillow。

Usage:
    # MobileCLIP2-S0
    CUDA_VISIBLE_DEVICES=0 python export_onnx.py \
        --checkpoint logs/mobileclip2/MobileCLIP2-S0_60.pth \
        --config     configs/person/vit_mobileclip2.yml \
        --output_dir deploy/mobileclip2/

    # CLIP-ReID ViT-B-16
    CUDA_VISIBLE_DEVICES=0 python export_onnx.py \
        --checkpoint logs/clipreid/ViT-B-16_60.pth \
        --config     configs/person/vit_clipreid.yml \
        --output_dir deploy/clipreid/

同一套 inference_onnx.py 通过不同的 deploy_config.json 加载任意模型。
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


logger = logging.getLogger("export_onnx")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)


# ============================================================================
# 模型无关的 ONNX 推理子图包装器
# 所有模型导出时都遵循: forward(images) -> features
# ============================================================================

class CLIPReIDInference(nn.Module):
    """CLIP-ReID (ViT-B-16 / RN50) 的推理子图.

    提取与训练时一致的推理特征:
      ViT-B-16: cat(feat_cls, feat_proj_cls) = [B, 768] + [B, 512] = [B, 1280]
      RN50:     cat(GAP(feat_last), GAP(feat), feat_proj[0]) = [B, 2048] + [B, 1024] = [B, 3072]

    输出不含 SIE（场外部署时通常无相机/视角信息），BNNeck 用 neck_feat='before'。
    """

    def __init__(self, model: "make_model_clipreid.build_transformer", neck_feat: str = "before"):
        super().__init__()
        self.image_encoder = model.image_encoder
        self.bottleneck = model.bottleneck
        self.bottleneck_proj = model.bottleneck_proj
        self.model_name = model.model_name
        self.in_planes = model.in_planes
        self.in_planes_proj = model.in_planes_proj
        self.neck_feat = neck_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == "RN50":
            feat_last_spatial, feat_spatial, feat_proj = self.image_encoder(x)
            feat_last = torch.nn.functional.avg_pool2d(
                feat_last_spatial, feat_last_spatial.shape[2:]
            ).view(x.shape[0], -1)
            feat = torch.nn.functional.avg_pool2d(
                feat_spatial, feat_spatial.shape[2:]
            ).view(x.shape[0], -1)
            feat_proj_out = feat_proj[0]
        else:  # ViT-B-16
            feat_last_spatial, feat_spatial, feat_proj = self.image_encoder(x, cv_emb=None)
            feat_last = feat_last_spatial[:, 0]    # CLS token
            feat = feat_spatial[:, 0]              # CLS token
            feat_proj_out = feat_proj[:, 0]        # CLS token

        if self.neck_feat == "after":
            feat_bn = self.bottleneck(feat)
            feat_proj_bn = self.bottleneck_proj(feat_proj_out)
            return torch.cat([feat_bn, feat_proj_bn], dim=1)
        else:
            return torch.cat([feat, feat_proj_out], dim=1)


class MobileCLIP2Inference(nn.Module):
    """MobileCLIP2-ReID 的推理子图.

    与 CLIPReIDInference 接口一致: forward(images) -> features.
    """

    def __init__(self, model: "make_model_mobileclip2.build_transformer", neck_feat: str = "before"):
        super().__init__()
        self.image_encoder = model.image_encoder
        self.bottleneck = model.bottleneck
        self.bottleneck_proj = model.bottleneck_proj
        self.neck_feat = neck_feat

        # FastViT: 多分支 -> 单分支 (仅推理需要)
        if hasattr(self.image_encoder, "reparameterize"):
            logger.info("[export] 正在重参数化 FastViT (多分支折叠为单 Conv2d) ...")
            self.image_encoder.reparameterize()
        else:
            logger.info("[export] 当前模型无 reparameterize() (ViT-B 变体)，跳过")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feat, feat_proj = self.image_encoder(x, cv_emb=None)

        if self.neck_feat == "after":
            feat_bn = self.bottleneck(feat)
            feat_proj_bn = self.bottleneck_proj(feat_proj)
            return torch.cat([feat_bn, feat_proj_bn], dim=1)
        else:
            return torch.cat([feat, feat_proj], dim=1)


# ============================================================================
# 模型加载
# ============================================================================

def load_model_for_export(checkpoint_path: str, config_path: str):
    """根据 config.MODEL.NAME 自动选择模型类，加载权重。

    Returns:
        (wrapper, meta) — wrapper 是推理子图，meta 是 deploy_config.json 内容
    """
    from config import cfg as _cfg

    _cfg.merge_from_file(config_path)
    _cfg.freeze()

    model_name: str = _cfg.MODEL.NAME
    neck_feat: str = _cfg.TEST.NECK_FEAT
    image_size = list(_cfg.INPUT.SIZE_TRAIN)  # [H, W]
    pixel_mean = list(_cfg.INPUT.PIXEL_MEAN)
    pixel_std = list(_cfg.INPUT.PIXEL_STD)
    feat_norm = _cfg.TEST.FEAT_NORM

    # --- 根据模型名选择构造方式 ---
    if model_name.startswith("MobileCLIP2"):
        from model.make_model_mobileclip2 import build_transformer as _build_mobileclip2

        logger.info(f"[export] 构建 MobileCLIP2 模型 (NAME={model_name}) ...")
        full_model = _build_mobileclip2(
            num_classes=1000,
            camera_num=1,
            view_num=1,
            cfg=_cfg,
        )
        wrapper: nn.Module = MobileCLIP2Inference(full_model, neck_feat=neck_feat)

    elif model_name in ("ViT-B-16", "RN50"):
        from model.make_model_clipreid import build_transformer as _build_clipreid

        logger.info(f"[export] 构建 CLIP-ReID 模型 (NAME={model_name}) ...")
        full_model = _build_clipreid(
            num_classes=1000,
            camera_num=1,
            view_num=1,
            cfg=_cfg,
        )
        wrapper = CLIPReIDInference(full_model, neck_feat=neck_feat)

    else:
        raise ValueError(
            f"[export] 不支持的 MODEL.NAME: {model_name}。"
            " 支持: ViT-B-16, RN50, MobileCLIP2-S0, MobileCLIP2-S2, MobileCLIP2-B"
        )

    # --- 加载 checkpoint ---
    logger.info(f"[export] 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("[export] checkpoint 格式不支持")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_state_dict[new_k] = v

    # 加载到完整模型（wrapper 共享 image_encoder / bottleneck）
    missing, unexpected = full_model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning(f"[export] 未加载的 keys ({len(missing)}): {missing[:20]}")
    if unexpected:
        logger.warning(f"[export] 未使用的 keys ({len(unexpected)}): {unexpected[:20]}")

    # 实例化 wrapper（加载了权重的 image_encoder/bottleneck 已被 wrapper 共享）
    if model_name.startswith("MobileCLIP2"):
        wrapper = MobileCLIP2Inference(full_model, neck_feat=neck_feat)
    else:
        wrapper = CLIPReIDInference(full_model, neck_feat=neck_feat)

    full_model.eval()
    wrapper.eval()

    # 推理子图的输出维度（从一次 forward 得到）
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size[0], image_size[1], dtype=torch.float32)
        out = wrapper(dummy)
        feat_dim = out.shape[1]

    meta = {
        "model_name": model_name,
        "image_size": image_size,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "neck_feat": neck_feat,
        "feat_norm": feat_norm,
        "feat_dim": feat_dim,
        "onnx_input_name": "images",
        "onnx_output_name": "features",
        "onnx_input_shape": f"[N, 3, {image_size[0]}, {image_size[1]}]",
        "onnx_input_dtype": "float32",
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
    }
    return wrapper, meta


# ============================================================================
# ONNX 导出 + 校验
# ============================================================================

@torch.no_grad()
def export_onnx(
    model: nn.Module,
    meta: dict,
    output_dir: str,
    dynamic_batch: bool = True,
    device: str = "cpu",
):
    os.makedirs(output_dir, exist_ok=True)

    # 模型名作为文件名后缀，便于区分
    safe_name = meta["model_name"].replace("-", "_").replace(".", "_")
    onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")
    json_path = os.path.join(output_dir, "deploy_config.json")

    H, W = meta["image_size"]
    dummy = torch.randn(1, 3, H, W, dtype=torch.float32).to(device)
    model = model.to(device)

    # --- reference torch output ---
    torch_out = model(dummy).cpu().numpy()
    logger.info(
        f"[export] torch 输出形状: {torch_out.shape}  "
        f"范围=[{torch_out.min():.4f}, {torch_out.max():.4f}]"
    )

    # --- dynamic_axes: batch 维度动态，H/W 静态 ---
    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}, "features": {0: "batch"}}
    else:
        dynamic_axes = None

    # --- export ---
    logger.info("[export] 正在 torch.onnx.export ...")
    t0 = time.time()
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
    )
    logger.info(f"[export] 导出完成 ({time.time() - t0:.1f}s) → {onnx_path}")

    # --- onnxruntime 数值校验 ---
    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"images": dummy.cpu().numpy()})[0]

        # cosine similarity
        cos = float(
            np.sum(torch_out * ort_out)
            / (np.linalg.norm(torch_out) * np.linalg.norm(ort_out) + 1e-12)
        )
        logger.info(f"[export] cosine(torch, onnx) = {cos:.6f}  (要求 > 0.999)")
        assert cos > 0.999, f"[export] 数值校验失败: cos={cos:.6f}"
        logger.info("[export] ✅ 数值校验通过")
    except ImportError:
        logger.warning("[export] onnxruntime 未安装，跳过校验。请 pip install onnxruntime")

    # --- 写 deploy_config.json ---
    meta["onnx_file"] = os.path.basename(onnx_path)
    meta["onnx_output_dim"] = int(torch_out.shape[1])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print()
    print("=" * 66)
    print(f"  ONNX 导出完成")
    print(f"  模型名称  : {meta['model_name']}")
    print(f"  模型文件  : {onnx_path}  ({size_mb:.1f} MB)")
    print(f"  配置文件  : {json_path}")
    print(f"  特征维度  : {torch_out.shape[1]}")
    print(f"  输入尺寸  :  N x 3 x {H} x {W}")
    print(f"  预处理    :  mean={pixel_mean}, std={pixel_std}")
    print("=" * 66)

    return onnx_path, json_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ReID ONNX 导出 — 支持 CLIP-ReID 和 MobileCLIP2-ReID"
    )
    parser.add_argument("--checkpoint", required=True, help="训练好的 .pth 文件")
    parser.add_argument(
        "--config", default="configs/person/vit_clipreid.yml",
        help="训练配置文件 (会自动根据 MODEL.NAME 选择模型)"
    )
    parser.add_argument("--output_dir", default="deploy/", help="输出目录")
    parser.add_argument("--no_dynamic_batch", action="store_true", help="固化 batch=1")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger.info(f"[export] 参数: {vars(args)}")

    device = torch.device(args.device)
    wrapper, meta = load_model_for_export(args.checkpoint, args.config)
    export_onnx(wrapper, meta, args.output_dir, dynamic_batch=not args.no_dynamic_batch, device=device)


if __name__ == "__main__":
    main()
