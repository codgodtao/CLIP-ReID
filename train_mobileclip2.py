"""
Training entry point for MobileCLIP2-based ReID.

Runs the two-stage training pipeline:
  Stage 1: Learn per-class prompt tokens (vision + text frozen, SupConLoss)
  Stage 2: Fine-tune vision encoder (text + prompt frozen, ID+Triplet+I2T loss)

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_mobileclip2.py \
      --config_file configs/person/vit_mobileclip2.yml
"""
import random
import os
import argparse

import torch
import numpy as np

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_mobileclip2 import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_mobileclip2_stage1 import do_train_stage1
from processor.processor_mobileclip2_stage2 import do_train_stage2


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileCLIP2 ReID Training")
    parser.add_argument(
        "--config_file",
        default="configs/person/vit_mobileclip2.yml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # Dataloader — reused from CLIP-ReID (preprocessing is config-driven)
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # --- Stage 1: Prompt learning ---
    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(
        optimizer_1stage,
        num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
        lr_min=cfg.SOLVER.STAGE1.LR_MIN,
        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
        noise_range=None,
    )

    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank,
    )

    # --- Stage 2: Vision encoder fine-tuning ---
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query,
        args.local_rank,
    )

    # ========================================================================
    # 训练结束 — 导出 ONNX
    # ========================================================================
    export_checkpoint = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_final.pth")
    torch.save(model.state_dict(), export_checkpoint)
    logger.info(f"保存最终权重: {export_checkpoint}")

    deploy_dir = os.path.join("deploy", cfg.MODEL.NAME.replace("-", "_"))
    logger.info(f"开始 ONNX 导出 -> {deploy_dir} ...")
    try:
        import export_onnx as _eo

        wrapper, meta = _eo.load_model_for_export(export_checkpoint, args.config_file)
        _eo.export_onnx(wrapper, meta, deploy_dir, dynamic_batch=True, device="cuda")
        logger.info("ONNX 导出完成")
    except Exception as _e:
        logger.warning(
            f"ONNX 导出失败: {_e}。"
            f"可稍后手动运行: python export_onnx.py "
            f"--checkpoint {export_checkpoint} "
            f"--config {args.config_file} "
            f"--output_dir {deploy_dir}/"
        )
