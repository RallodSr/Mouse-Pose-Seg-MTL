"""
Mouse Pose & Segmentation — Multi-Task Learning
================================================
Single CLI entry point for the MTL model.

Commands
--------
  prepare      Convert Label Studio export → dataset.json
  train        Train HybridMTLNet (supports custom weight ratios)
  eval         Evaluate a trained checkpoint on the test set
  experiment   Run a train-and-compare study (weight-ratio | single-task)

Examples
--------
  python main.py prepare
  python main.py train
  python main.py train --seg-weight 1 --pose-weight 40 --run-name baseline_1_40
  python main.py eval --weights models/checkpoints/model_final.pth
  python main.py experiment --type single-task
  python main.py experiment --type weight-ratio --eval-only

Baselines (separate, in baselines/): maskrcnn.py, yolo/
"""
import argparse
import copy

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG


def cmd_prepare(args):
    from src.data.prepare import prepare_dataset
    prepare_dataset(
        json_path=DATA_CFG.label_studio_json,
        images_dir=DATA_CFG.images_dir,
        masks_dir=DATA_CFG.masks_dir,
        output_json=DATA_CFG.json_path,
    )


def cmd_train(args):
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
    from src.evaluation.evaluator import Evaluator

    miou, pck = Evaluator(DATA_CFG, MODEL_CFG, TRAIN_CFG).evaluate(args.weights, task=args.task)
    print(f"Test mIoU : {miou:.4f}")
    print(f"Test PCK  : {pck:.4f}")


def cmd_benchmark(args):
    from src.evaluation.benchmark import Benchmark
    Benchmark(MODEL_CFG, TRAIN_CFG, input_size=DATA_CFG.target_size, runs=args.runs).run()


def cmd_experiment(args):
    from src.experiments import WeightRatioExperiment, SingleTaskAblation

    cls = {"weight-ratio": WeightRatioExperiment, "single-task": SingleTaskAblation}[args.type]
    exp = cls(DATA_CFG, MODEL_CFG, TRAIN_CFG)

    results = exp.eval_only() if args.eval_only else exp.run_all()
    if results:
        exp.print_table(results)
        if not args.no_plot and hasattr(exp, "plot"):
            exp.plot(results)
    else:
        print("No results to display.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mouse Pose & Seg MTL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Label Studio JSON → dataset.json")

    p_train = sub.add_parser("train", help="Train MTL model")
    p_train.add_argument("--seg-weight",  type=float, default=None,
                         help="Override loss_seg_weight  (default: from config.py)")
    p_train.add_argument("--pose-weight", type=float, default=None,
                         help="Override loss_pose_weight (default: from config.py)")
    p_train.add_argument("--run-name",    type=str,   default=None,
                         help="Sub-folder name under models/checkpoints/")

    p_eval = sub.add_parser("eval", help="Evaluate MTL on test set")
    p_eval.add_argument("--weights", default=f"{TRAIN_CFG.output_dir}/model_final.pth")
    p_eval.add_argument("--task", choices=["joint", "seg", "pose"], default="joint",
                        help="Instance-matching mode (pose = heatmap-based)")

    p_exp = sub.add_parser("experiment", help="Run a train-and-compare study")
    p_exp.add_argument("--type", choices=["weight-ratio", "single-task"], required=True)
    p_exp.add_argument("--eval-only", action="store_true",
                       help="Skip training — only evaluate saved checkpoints")
    p_exp.add_argument("--no-plot", action="store_true", help="Skip chart generation")

    p_bench = sub.add_parser("benchmark", help="Measure params/FLOPs/latency (efficiency)")
    p_bench.add_argument("--runs", type=int, default=100, help="Timed iterations")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    dispatch = {
        "prepare":    cmd_prepare,
        "train":      cmd_train,
        "eval":       cmd_eval,
        "experiment": cmd_experiment,
        "benchmark":  cmd_benchmark,
    }
    dispatch[args.command](args)
