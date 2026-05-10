"""
Mouse Pose & Segmentation — Multi-Task Learning
================================================
CLI entry point for all major operations.

Commands
--------
  prepare     Convert Label Studio export → dataset.json
  convert     Convert dataset.json → YOLO format (pose / seg)
  train       Train the MTL model (supports custom weight ratios)
  eval        Evaluate MTL model on test set
  eval-yolo   Evaluate YOLO baseline models

Examples
--------
  python main.py prepare
  python main.py convert --task pose
  python main.py train
  python main.py train --seg-weight 1 --pose-weight 40 --run-name baseline_1_40
  python main.py eval --weights models/checkpoints/model_final.pth
  python run_experiments.py            # รันทุก weight config แล้วเปรียบเทียบ
  python run_experiments.py --eval-only
"""
import argparse
import json
import sys

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG


def cmd_prepare(args):
    from src.data.prepare import prepare_dataset
    prepare_dataset(
        json_path=DATA_CFG.label_studio_json,
        images_dir=DATA_CFG.images_dir,
        masks_dir=DATA_CFG.masks_dir,
        output_json=DATA_CFG.json_path,
    )


def cmd_convert(args):
    from src.data.prepare import convert_yolo_pose, convert_yolo_seg
    if args.task == "pose":
        convert_yolo_pose(DATA_CFG.json_path)
    elif args.task == "seg":
        convert_yolo_seg(DATA_CFG.json_path)
    else:
        print("--task must be 'pose' or 'seg'")
        sys.exit(1)


def cmd_train(args):
    import copy
    from src.training.trainer import Trainer

    cfg = copy.copy(TRAIN_CFG)
    if args.seg_weight is not None:
        cfg.loss_seg_weight = args.seg_weight
    if args.pose_weight is not None:
        cfg.loss_pose_weight = args.pose_weight
    if args.run_name:
        cfg.output_dir = f"models/checkpoints/{args.run_name}"

    print(f"seg_weight={cfg.loss_seg_weight}  pose_weight={cfg.loss_pose_weight}  → {cfg.output_dir}")
    Trainer(DATA_CFG, MODEL_CFG, cfg).run()


def cmd_eval(args):
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import MouseMTLDataset
    from src.models.mtl_net import HybridMTLNet
    from src.evaluation.metrics import calculate_miou, calculate_pck

    device = torch.device(TRAIN_CFG.device)
    with open(DATA_CFG.json_path) as f:
        data = json.load(f)

    test_ds = MouseMTLDataset(data["test"], DATA_CFG.target_size, DATA_CFG.sigma)
    test_dl = DataLoader(test_ds, batch_size=TRAIN_CFG.batch_size, shuffle=False,
                         num_workers=TRAIN_CFG.num_workers)

    model = HybridMTLNet(MODEL_CFG.num_classes, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    total_miou = total_pck = 0.0
    with torch.no_grad():
        for imgs, masks, hms in test_dl:
            imgs, masks, hms = imgs.to(device), masks.to(device), hms.to(device)
            pred_seg, pred_pose = model(imgs)
            total_miou += calculate_miou(pred_seg, masks)
            total_pck += calculate_pck(pred_pose, hms, TRAIN_CFG.pck_threshold)

    n = len(test_dl)
    print(f"Test mIoU : {total_miou/n:.4f}")
    print(f"Test PCK  : {total_pck/n:.4f}")


def cmd_eval_yolo(args):
    if args.task == "seg":
        from src.evaluation.yolo import evaluate_yolo_miou
        evaluate_yolo_miou(DATA_CFG.json_path, args.weights, DATA_CFG.target_size)
    elif args.task == "pose":
        from src.evaluation.yolo import evaluate_yolo_pck
        evaluate_yolo_pck(DATA_CFG.json_path, args.weights, DATA_CFG.target_size, TRAIN_CFG.pck_threshold)
    else:
        print("--task must be 'pose' or 'seg'")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mouse Pose & Seg MTL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Label Studio JSON → dataset.json")

    p_conv = sub.add_parser("convert", help="dataset.json → YOLO format")
    p_conv.add_argument("--task", choices=["pose", "seg"], required=True)

    p_train = sub.add_parser("train", help="Train MTL model")
    p_train.add_argument("--seg-weight",  type=float, default=None,
                         help="Override loss_seg_weight  (default: from config.py)")
    p_train.add_argument("--pose-weight", type=float, default=None,
                         help="Override loss_pose_weight (default: from config.py)")
    p_train.add_argument("--run-name",    type=str,   default=None,
                         help="Sub-folder name under models/checkpoints/")

    p_eval = sub.add_parser("eval", help="Evaluate MTL on test set")
    p_eval.add_argument("--weights", default=f"{TRAIN_CFG.output_dir}/model_final.pth")

    p_yolo = sub.add_parser("eval-yolo", help="Evaluate YOLO baseline")
    p_yolo.add_argument("--task", choices=["pose", "seg"], required=True)
    p_yolo.add_argument("--weights", required=True)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    dispatch = {
        "prepare": cmd_prepare,
        "convert": cmd_convert,
        "train": cmd_train,
        "eval": cmd_eval,
        "eval-yolo": cmd_eval_yolo,
    }
    dispatch[args.command](args)
