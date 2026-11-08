"""Command-line entry point for the MindSpore LDEPrompt implementation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LDEPrompt on MindSpore/CANN")
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--exp", default="mindspore", help="Experiment/output name")
    parser.add_argument("--output-dir", default="", help="Explicit output directory")
    parser.add_argument("--dataset-root", default="", help="Override dataset_root")
    parser.add_argument("--device", default="npu:0", help="npu:0, npu:1, Ascend:0, or cpu")
    parser.add_argument(
        "--device-target",
        default="auto",
        choices=["auto", "ascend", "npu", "cann", "cpu", "Ascend", "NPU", "CANN", "CPU"],
    )
    parser.add_argument("--ms-mode", default="pynative", choices=["pynative", "graph"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="MindSpore ViT backbone .ckpt produced by scripts/convert_jax_vit.py",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override every configured seed")
    return parser


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
        force=True,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .config import load_config, resolve_seeds, validate_config
    from .data import ContinualDataManager
    from .device import describe_device, setup_context
    from .trainer import LDEPromptLearner, jsonable

    config = load_config(args.config)
    if args.dataset_root:
        config["dataset_root"] = args.dataset_root
    if args.num_workers is not None:
        config["num_workers"] = int(args.num_workers)
    if args.max_tasks is not None:
        config["max_tasks"] = int(args.max_tasks)
    if args.backbone_checkpoint is not None:
        config["backbone_checkpoint"] = args.backbone_checkpoint
    validate_config(config)
    seeds = resolve_seeds(config, args.seed)

    base_output = Path(args.output_dir) if args.output_dir else Path("runs") / args.exp
    base_output.mkdir(parents=True, exist_ok=True)
    configure_logging(base_output)
    all_metrics = []
    for seed in seeds:
        run_config = dict(config)
        run_config.update(
            {
                "seed": seed,
                "device": args.device,
                "device_target": args.device_target,
                "ms_mode": args.ms_mode,
            }
        )
        runtime = setup_context(run_config)
        runtime["command"] = " ".join(sys.argv)
        output_dir = base_output if len(seeds) == 1 else base_output / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(output_dir)
        logging.info("%s", describe_device(runtime))
        logging.info("Python %s | config=%s", sys.version.split()[0], args.config)
        data_manager = ContinualDataManager(run_config)
        logging.info(
            "Dataset=%s tasks=%d increments=%s classes=%d",
            run_config["dataset"],
            data_manager.num_tasks,
            data_manager.increments,
            data_manager.nb_classes,
        )
        learner = LDEPromptLearner(
            run_config, data_manager, output_dir=output_dir, runtime_info=runtime
        )
        metrics = learner.run()
        with open(output_dir / "run.json", "w", encoding="utf-8") as handle:
            json.dump(
                jsonable({
                    "runtime": runtime,
                    "metrics": metrics,
                }),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        all_metrics.append(metrics)
        logging.info(
            "Finished seed=%d avg=%.2f last=%.2f forgetting=%.2f",
            seed,
            metrics["average_accuracy"],
            metrics["last_accuracy"],
            metrics["forgetting"],
        )
    logging.info("Completed %d seed(s). Results under %s", len(all_metrics), base_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
