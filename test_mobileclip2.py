"""
Inference entry point for MobileCLIP2-based ReID.

Loads a trained checkpoint and runs evaluation. By default, the FastViT
multi-branched structure is reparameterized into single-branch convs for
faster inference (use --no_reparam to disable).

Usage:
  CUDA_VISIBLE_DEVICES=0 python test_mobileclip2.py \
      --config_file configs/person/vit_mobileclip2.yml \
      --test_weight path/to/MobileCLIP2-S0_60.pth
"""
import os
import argparse

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_mobileclip2 import make_model
from processor.processor_mobileclip2_stage2 import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileCLIP2 ReID Inference")
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
    parser.add_argument(
        "--no_reparam",
        action="store_true",
        help="Disable FastViT reparameterization (keep multi-branched structure)",
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    reparam = not args.no_reparam

    if cfg.DATASETS.NAMES == "VehicleID":
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg, model, val_loader, num_query, reparameterize=reparam)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info(
            "sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(
                all_rank_1.sum() / 10.0, all_rank_5.sum() / 10.0
            )
        )
    else:
        do_inference(cfg, model, val_loader, num_query, reparameterize=reparam)
