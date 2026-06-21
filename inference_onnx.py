"""场外 ONNX 推理 — 只需要 numpy + Pillow + onnxruntime。

完全独立于 PyTorch / timm / mobileclip，可部署到任意设备。

Usage:
    python inference_onnx.py \
        --onnx          deploy/mobileclip2-reid.onnx \
        --config        deploy/deploy_config.json \
        --image         assets/query/0001_c1s1_000301_00.jpg \
        --image_dir     assets/gallery/ \
        --top_k         10 \
        --save_vis      results/vis/

部署步骤:
    1. 从训练机复制 deploy/mobileclip2-reid.onnx + deploy/deploy_config.json
    2. 目标机只需:
           pip install onnxruntime  (或 onnxruntime-gpu)
           pip install numpy Pillow
    3. 预处理完全在 numpy/Pillow 里实现，与训练时一致
"""
import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("onnx_inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================================
# 预处理 — numpy 实现，与训练时 cfg.INPUT 完全一致
# ============================================================================
def letterbox_resize(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (124, 116, 104),
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """等比例缩放 + 居中填充 (LetterBox)。

    与 training 时的 T.Resize + T.Pad + T.RandomCrop 不同，
    推理时只做等比例 resize + pad，保证输入尺寸严格一致。

    Returns:
        (resized_pil, scale, pad_offset)
        - scale: 原始图像相对于缩放后尺寸的比例
        - pad_offset: (pad_top, pad_left)
    """
    target_h, target_w = target_size
    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, target_h / orig_h)

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    # 创建 target_size 灰色画布
    canvas = Image.new("RGB", (target_w, target_h), fill_color)
    # 居中粘贴
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, (pad_top, pad_left)


class Preprocessor:
    """MobileCLIP2 ReID 预处理 (纯 numpy / Pillow)。"""

    def __init__(
        self,
        image_size: Tuple[int, int],
        pixel_mean: List[float],
        pixel_std: List[float],
        resize_mode: str = "letterbox",
    ):
        self.image_size = image_size  # (H, W)
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.array(pixel_std, dtype=np.float32).reshape(1, 1, 3)
        self.resize_mode = resize_mode

    def __call__(self, image: Image.Image) -> np.ndarray:
        """PIL Image → CHW float32 numpy (已归一化)。"""
        if self.resize_mode == "letterbox":
            image = letterbox_resize(image, self.image_size)[0]
        elif self.resize_mode == "直接resize":
            image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        else:
            raise ValueError(f"未知 resize_mode: {self.resize_mode}")

        # PIL -> numpy [H, W, C], uint8
        arr = np.asarray(image, dtype=np.uint8)

        # ToTensor: [0, 255] -> [0, 1]
        arr = arr.astype(np.float32) / 255.0

        # Normalize: (x - mean) / std
        arr = (arr - self.pixel_mean) / self.pixel_std

        # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
        return arr

    def batch(self, images: List[Image.Image]) -> np.ndarray:
        """批量预处理。"""
        return np.stack([self(img) for img in images], axis=0)


# ============================================================================
# 特征提取
# ============================================================================
class ReIDEngine:
    """ONNX Runtime 推理引擎。"""

    def __init__(self, onnx_path: str, config_path: Optional[str] = None, **overrides):
        t0 = time.time()
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime 未安装。请运行: pip install onnxruntime (或 onnxruntime-gpu)"
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT_ENABLE_ALL 自动选择最优 provider (CPU / CUDA / TensorRT / CoreML...)
        self.session = ort.InferenceSession(onnx_path, sess_opts)
        logger.info(
            f"[engine] onnxruntime providers: {self.session.get_providers()}, "
            f"耗时 {time.time() - t0:.1f}s"
        )

        # 加载/合并配置
        if config_path and os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}

        # CLI 覆盖
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

        self.meta = cfg
        self.input_name = cfg.get("onnx_input_name", "images")
        self.output_name = cfg.get("onnx_output_name", "features")
        self.feat_norm = cfg.get("feat_norm", "yes")
        self.feat_dim = cfg.get("onnx_output_dim", None)

    def extract(self, images: np.ndarray) -> np.ndarray:
        """批量特征提取。

        Args:
            images: [B, 3, H, W] float32, 已归一化

        Returns:
            features: [B, D] float32, L2 归一化 (feat_norm='yes')
        """
        out = self.session.run(
            [self.output_name],
            {self.input_name: images.astype(np.float32)},
        )[0]
        if self.feat_norm == "yes":
            out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
        return out

    def extract_single(self, image: np.ndarray) -> np.ndarray:
        """单张图像推理。"""
        return self.extract(image[np.newaxis, ...])[0]

    def extract_from_pil(self, preprocessor: Preprocessor, image: Image.Image) -> np.ndarray:
        """PIL Image → 特征向量 (含预处理)。"""
        arr = preprocessor(image)[np.newaxis, ...]
        return self.extract(arr)[0]


# ============================================================================
# 搜索 / 评测工具
# ============================================================================
def search_topk(
    query_feat: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_cids: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """余弦距离 Top-K 检索。

    Args:
        query_feat: [D] 查询特征
        gallery_feats: [N, D] 底库特征
        gallery_pids: [N] person ID
        gallery_cids: [N] camera ID
        k: 返回数量

    Returns:
        (distances, pids, cids) 各 k 个
    """
    dists = 1.0 - query_feat @ gallery_feats.T   # cosine distance
    indices = np.argsort(dists)[:k]
    return dists[indices], gallery_pids[indices], gallery_cids[indices]


def build_gallery(
    engine: ReIDEngine,
    preprocessor: Preprocessor,
    image_dir: str,
    recursive: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """扫描目录下所有 .jpg/.png 构建底库特征。

    底库图像命名约定: {pid}_{cam}_{...}.jpg
    即第一个下划线前是 pid，第二个下划线前是 cam。

    Returns:
        (features, paths, pids, cids)
    """
    paths = sorted(Path(image_dir).rglob("*.jpg" if recursive else "*.jpg"))
    paths = [p for p in paths if not p.name.startswith(".")] + \
            sorted(Path(image_dir).rglob("*.png" if recursive else "*.png"))
    paths = [p for p in paths if not p.name.startswith(".")]

    logger.info(f"[gallery] 扫描到 {len(paths)} 张图像 ...")

    feats = []
    pids, cids = [], []
    batch_size = 32

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_arr = preprocessor.batch(batch_imgs)
        batch_feats = engine.extract(batch_arr)

        for p, f in zip(batch_paths, batch_feats):
            feats.append(f)
            name = p.stem  # "0001_c1s1_000301_00" -> "0001_c1s1_000301_00"
            parts = name.split("_")
            try:
                pid = int(parts[0])
            except (ValueError, IndexError):
                pid = hash(name) & 0x7FFFFFFF
            try:
                cid = int(parts[1][1:]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                cid = 0
            pids.append(pid)
            cids.append(cid)

    feats = np.stack(feats, axis=0).astype(np.float32)
    pids = np.array(pids, dtype=np.int32)
    cids = np.array(cids, dtype=np.int32)
    path_strs = [str(p) for p in paths]

    logger.info(f"[gallery] 底库特征形状: {feats.shape}")
    return feats, path_strs, pids, cids


def evaluate(
    query_dir: str,
    gallery_dir: str,
    engine: ReIDEngine,
    preprocessor: Preprocessor,
    top_k: int = 50,
) -> Tuple[float, np.ndarray]:
    """在 query / gallery 上跑 mAP + CMC。

    返回 (mAP, cmc_curve)
    """
    gallery_feats, gallery_paths, gallery_pids, gallery_cids = build_gallery(
        engine, preprocessor, gallery_dir, recursive=True
    )
    query_paths = sorted(Path(query_dir).rglob("*.jpg"))
    query_paths = [p for p in query_paths if not p.name.startswith(".")]
    logger.info(f"[eval] Query: {len(query_paths)}, Gallery: {len(gallery_paths)}")

    aps = []
    cmc = np.zeros(top_k, dtype=np.float32)

    for qp in query_paths:
        q_img = Image.open(qp).convert("RGB")
        q_feat = engine.extract_from_pil(preprocessor, q_img)

        # 底库搜索 (排除同相机同pid的干扰项)
        parts = qp.stem.split("_")
        q_pid = int(parts[0])
        q_cid = int(parts[1][1:]) if len(parts) > 1 else 0

        dists = 1.0 - q_feat @ gallery_feats.T
        indices = np.argsort(dists)

        # CMC@1
        for j, idx in enumerate(indices[:top_k]):
            if gallery_pids[idx] == q_pid and gallery_cids[idx] != q_cid:
                cmc[j:] += 1
                break

        # AP
        relevant = (gallery_pids == q_pid) & (gallery_cids != q_cid)
        if relevant.sum() == 0:
            continue
        ranked_relevant = relevant[indices]
        tp = ranked_relevant.sum()
        fp = (~ranked_relevant).sum()
        prec = np.cumsum(ranked_relevant) / (np.arange(len(ranked_relevant)) + 1)
        recall = np.cumsum(ranked_relevant) / tp
        ap = (recall[ranked_relevant] * prec[ranked_relevant]).sum()
        aps.append(ap)

    cmc /= max(len(query_paths), 1)
    mAP = np.mean(aps) if aps else 0.0

    logger.info(f"[eval] mAP={mAP:.1%}")
    for r in [1, 5, 10, 20]:
        if r <= top_k:
            logger.info(f"  Rank-{r}: {cmc[r-1]:.1%}")
    return mAP, cmc


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MobileCLIP2 ReID ONNX 推理")
    # --- 模型 ---
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径")
    parser.add_argument("--config", default="", help="deploy_config.json 路径 (可选)")
    # --- 图像 ---
    parser.add_argument("--image", default="", help="单张图像路径")
    parser.add_argument("--image_dir", default="", help="底库目录")
    parser.add_argument("--query_dir", default="", help="查询目录 (用于评测)")
    # --- 预处理覆盖 (来自 deploy_config.json，也可在命令行覆盖) ---
    parser.add_argument("--image_size", default="", help="H,W (如 256,128)")
    parser.add_argument("--pixel_mean", default="", help="R,G,B 均值 (如 0,0,0)")
    parser.add_argument("--pixel_std", default="", help="R,G,B 标准差 (如 1,1,1)")
    # --- 搜索/评测 ---
    parser.add_argument("--top_k", type=int, default=10, help="返回 top_k 结果")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--save_vis", default="", help="可视化结果保存目录")
    args = parser.parse_args()

    # --- 预处理配置 ---
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    image_size = tuple(map(int, args.image_size.split(","))) if args.image_size else tuple(cfg.get("image_size", [256, 128]))
    pixel_mean = list(map(float, args.pixel_mean.split(","))) if args.pixel_mean else cfg.get("pixel_mean", [0.0, 0.0, 0.0])
    pixel_std = list(map(float, args.pixel_std.split(","))) if args.pixel_std else cfg.get("pixel_std", [1.0, 1.0, 1.0])

    preprocessor = Preprocessor(image_size=image_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    engine = ReIDEngine(args.onnx, args.config)

    # --- 单图模式 ---
    if args.image:
        img = Image.open(args.image).convert("RGB")
        feat = engine.extract_from_pil(preprocessor, img)
        print(f"\n图像: {args.image}")
        print(f"特征维度: {feat.shape}  范围=[{feat.min():.4f}, {feat.max():.4f}]")

        # 底库搜索
        if args.image_dir:
            gallery_feats, gallery_paths, gallery_pids, gallery_cids = build_gallery(
                engine, preprocessor, args.image_dir
            )
            dists, pids, cids = search_topk(feat, gallery_feats, gallery_pids, gallery_cids, k=args.top_k)
            print(f"\n{'排名':<6} {'距离':<10} {'PID':<8} {'相机':<6} {'路径'}")
            print("-" * 80)
            for rank, (d, p, c, path) in enumerate(zip(dists, pids, cids, gallery_paths), 1):
                print(f"{rank:<6} {d:<10.4f} {p:<8} {c:<6} {Path(path).name}")
        return

    # --- 评测模式 ---
    if args.query_dir and args.image_dir:
        mAP, cmc = evaluate(args.query_dir, args.image_dir, engine, preprocessor, top_k=args.top_k)
        print(f"\n最终 mAP: {mAP:.1%}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
