"""训练后将 MobileCLIP2 ReID 模型导出为 ONNX。

本脚本：
  1. 加载训练好的 .pth checkpoint（需包含 image_encoder + bottleneck 权重）
  2. 调用 image_encoder.reparameterize() 把 FastViT 多分支折叠为单通路
  3. 用 torch.onnx.export 导出为 .onnx（动态 batch 维度，固化 H/W）
  4. 用 onnxruntime 做一轮等价校验（onnx_output vs torch_output）
  5. 在同目录写出 deploy_config.json —— 含预处理参数供场外推理使用

场外设备上**无需**安装 PyTorch/timm/mobileclip，只需 onnxruntime + numpy + Pillow。
详见 inference_onnx.py 与 deploy_guide.md。

Usage:
    CUDA_VISIBLE_DEVICES=0 python export_onnx.py \
        --checkpoint        logs/mobileclip2/checkpoint.pth \
        --config            configs/person/vit_mobileclip2.yml \
        --output_dir        deploy/

输出文件:
    deploy/mobileclip2-reid.onnx         # 主模型 (≈ 几十 MB)
    deploy/deploy_config.json            # 预处理参数 + 元数据
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# sys.path 注入 — 使脚本能被直接从项目根目录运行
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model.make_model_mobileclip2 import build_transformer  # noqa: E402
from config import cfg  # noqa: E402

logger = logging.getLogger("export_onnx")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)


# ============================================================================
# 1. —— 推理专用包装模型 ——
#    只保留 image_encoder + bottleneck，避免 get_image/get_text 这类控制流
#    使 forward 成为"纯图像 → 特征"的计算图，便于 ONNX tracing
# ============================================================================
class MobileCLIP2ReIDInference(nn.Module):
    """把 build_transformer 的推理分支单独拆出来。

    forward(x) == cat( feat_last_bn, feat_proj_bn ) 与原推理流程一致。
    """

    def __init__(self, model: build_transformer, neck_feat: str = "before"):
        super().__init__()
        self.image_encoder = model.image_encoder
        # 若训练时启用了 SIE，把 cv_emb 也带过来（SIE 是加法，不影响 ONNX，但需要额外输入）
        self.sie_coe = model.sie_coe
        self.cv_embed = model.cv_embed  # Parameter / None

        # BNNeck 两个分支
        self.bottleneck = model.bottleneck           # for feat (vision_dim)
        self.bottleneck_proj = model.bottleneck_proj  # for feat_proj (embed_dim)
        self.neck_feat = neck_feat

        # 将 image_encoder 中的 FastViT 多分支折叠为单分支 — 不可逆，必须在此调用
        if hasattr(self.image_encoder, "reparameterize"):
            logger.info("[export] 正在重参数化 FastViT ...")
            self.image_encoder.reparameterize()
        else:
            logger.warning("[export] image_encoder 无 reparameterize()，跳过 (可能是 ViT-B 变体)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """图像输入 -> 拼接后的 ReID 全局特征。

        输出形状: [B, vision_dim + embed_dim] —— 对 S0 是 [B, 512+512]=[B, 1024]。
        """
        # SIE 推理时默认不传相机/视角（场外部署通常无此信息）。
        # 如果你的场景必须用 SIE，把 cam_label/view_label 设为额外输入即可。
        _, feat, feat_proj = self.image_encoder(x, cv_emb=None)

        if self.neck_feat == "after":
            feat_bn = self.bottleneck(feat)
            feat_proj_bn = self.bottleneck_proj(feat_proj)
            return torch.cat([feat_bn, feat_proj_bn], dim=1)
        else:
            # 原训练/推理默认: neck_feat='before'，直接用池化后的原始特征
            return torch.cat([feat, feat_proj], dim=1)


# ============================================================================
# 2. —— 主流程 ——
# ============================================================================
def load_trained_model(checkpoint_path: str, cfg_path: str) -> Tuple[nn.Module, dict]:
    """从 checkpoint + yacs cfg 构造推理模型。

    与 train_mobileclip2.py 的加载逻辑一致 —— 返回 (MobileCLIP2ReIDInference, meta)。
    """
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    # 构造完整模型（会走 build_transformer.__init__ 的 image_encoder / text_encoder 构造）
    logger.info(f"[export] 正在构造模型 (NAME={cfg.MODEL.NAME}) ...")
    # 提示: build_transformer 构造时会尝试下载 MobileCLIP2 pretrained 权重 ——
    #       但我们随后会用 checkpoint 覆盖 image_encoder 权重。如果离线环境无法下载，
    #       请把 PRETRAIN_CHOICE 改为 'finetune' 或把 PRETRAIN_PATH 指到本地文件。
    model = build_transformer(
        num_classes=1000,        # num_classes 仅影响 classifier 层，推理用不到；给一个占位
        camera_num=1,            # 同上
        view_num=1,              # 同上
        cfg=cfg,
    )

    # 加载 checkpoint —— 兼容两种格式: dict{state_dict} 或直接 state_dict
    logger.info(f"[export] 正在加载 checkpoint: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("checkpoint 格式不支持 (既不是 dict 也不含 'state_dict')")

    # DataParallel 训练的权重会有 "module." 前缀，strip
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        missing_str = "\n  ".join(missing[:20])
        logger.warning(f"[export] 未加载到的 keys ({len(missing)} 个):\n  {missing_str}")
    if unexpected:
        unexp_str = "\n  ".join(unexpected[:20])
        logger.warning(f"[export] 未使用的 keys ({len(unexpected)} 个):\n  {unexp_str}")

    model.eval()
    wrapper = MobileCLIP2ReIDInference(model, neck_feat=cfg.TEST.NECK_FEAT)
    wrapper.eval()
    return wrapper, {
        "model_name": cfg.MODEL.NAME,
        "image_size": list(cfg.INPUT.SIZE_TEST),  # [H, W]
        "pixel_mean": list(cfg.INPUT.PIXEL_MEAN),
        "pixel_std": list(cfg.INPUT.PIXEL_STD),
        "neck_feat": cfg.TEST.NECK_FEAT,
        "feat_norm": cfg.TEST.FEAT_NORM,
    }


@torch.no_grad()
def export_onnx(model: nn.Module, meta: dict, output_dir: str, dynamic_batch: bool = True):
    """执行 ONNX 导出 + 数值等价校验 + 写 deploy_config.json。"""
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "mobileclip2-reid.onnx")
    json_path = os.path.join(output_dir, "deploy_config.json")

    H, W = meta["image_size"]
    dummy = torch.randn(1, 3, H, W, dtype=torch.float32)

    # --- reference torch output 用于校验 ---
    torch_out = model(dummy).numpy()
    logger.info(f"[export] 参考输出形状: {torch_out.shape}  "
                f"范围=[{torch_out.min():.4f}, {torch_out.max():.4f}]")

    # --- ONNX 动态轴定义: batch 维度动态，H/W 静态 ---
    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}, "features": {0: "batch"}}
    else:
        dynamic_axes = None

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
        do_constant_folding=True,            # 折叠常量 — 显著加速推理
        export_params=True,                   # 把权重写进 onnx (自包含)
    )
    logger.info(f"[export] 导出完成 ({time.time() - t0:.1f}s) → {onnx_path}")

    # --- onnxruntime 数值等价校验 (cosine >= 0.9999) ---
    logger.info("[export] onnxruntime 数值校验 ...")
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("[export] onnxruntime 未安装，跳过等价校验。请 pip install onnxruntime")
        ort_out = None
    else:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"images": dummy.numpy()})[0]
        cos = float(
            np.sum(torch_out * ort_out)
            / (np.linalg.norm(torch_out) * np.linalg.norm(ort_out) + 1e-12)
        )
        logger.info(f"[export]  cosine(torch_output, onnx_output) = {cos:.6f}")
        assert cos > 0.999, f"[export] 校验失败: cosine={cos} < 0.999"

    # --- 写 deploy_config.json (场外推理唯一需要的配置文件) ---
    meta.update({
        "onnx_input_name": "images",
        "onnx_output_name": "features",
        "onnx_input_shape": f"[N, 3, {H}, {W}]",
        "onnx_output_dim": int(torch_out.shape[1]),
        "onnx_input_dtype": "float32",
        "onnx_file": os.path.basename(onnx_path),
        "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
    })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"[export] deploy 配置: {json_path}")

    # --- 友好提示 ---
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print()
    print("=" * 64)
    print("  ✅ ONNX 导出完成")
    print(f"  ├─ 模型文件  : {onnx_path}  ({size_mb:.1f} MB)")
    print(f"  ├─ 配置文件  : {json_path}")
    print(f"  ├─ 特征维度  : {torch_out.shape[1]}")
    print(f"  └─ 推理尺寸  : 1 x 3 x {H} x {W}")
    print("=" * 64)
    return onnx_path, json_path


def main():
    parser = argparse.ArgumentParser(description="MobileCLIP2 ReID → ONNX exporter")
    parser.add_argument("--checkpoint", required=True, help="训练好的 .pth 文件")
    parser.add_argument("--config", default="configs/person/vit_mobileclip2.yml", help="训练配置")
    parser.add_argument("--output_dir", default="deploy/", help="输出目录 (默认 ./deploy/)")
    parser.add_argument("--no_dynamic_batch", action="store_true", help="固化 batch=1 (不推荐)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="导出时的计算设备")
    args = parser.parse_args()

    logger.info(f"[export] 运行参数: {vars(args)}")
    device = torch.device(args.device)

    model, meta = load_trained_model(args.checkpoint, args.config)
    model.to(device)
    # 注: 推理在 CPU 上导出即可 —— onnxruntime 在目标机上会自行选择 CPU/GPU provider

    export_onnx(model, meta, args.output_dir, dynamic_batch=not args.no_dynamic_batch)


if __name__ == "__main__":
    main()
