"""Configuration loading and validation for the MindSpore implementation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": "synthetic",
    "dataset_root": "data",
    "download": True,
    "shuffle_classes": True,
    "init_cls": 10,
    "increment": 10,
    "max_tasks": 0,
    "img_size": 224,
    "patch_size": 16,
    "in_chans": 3,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "batch_size": 24,
    "tuned_epoch": 20,
    "crct_epochs": 0,
    "ca_lr": 0.005,
    "ca_storage_efficient_method": "covariance",
    "ca_samples_per_class": 120,
    "n_centroids": 10,
    "init_lr": 0.001,
    "min_lr": 1e-5,
    "weight_decay": 0.0,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "init_milestones": [10, 15],
    "init_lr_decay": 0.5,
    "seed": 1993,
    "pretrained": False,
    "backbone_checkpoint": "",
    "freeze_backbone": True,
    "pool_size": 10,
    "length": 5,
    "top_k": 5,
    "num_placeholder": 3,
    "same_key_value": False,
    "batchwise_prompt": True,
    "use_g_prompt": True,
    "prompt_init": "uniform",
    "prompt_key_init": "uniform",
    "diversity_loss_coeff": 0.1,
    "placeholder_loss_coeff": 0.1,
    "importance_normalization": "zscore",
    "importance_max_batches": 0,
    "train_augment": True,
    "eval_resize": True,
    "normalize": False,
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
    "num_workers": 0,
    "synthetic_classes": 12,
    "synthetic_samples_per_class": 12,
    "synthetic_eval_samples_per_class": 4,
    "synthetic_noise": 0.08,
    "synthetic_seed": 2026,
    "max_samples_per_class": 0,
    "eval_samples_per_class": 0,
    "ca_batch_size": 256,
    "save_checkpoint": True,
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON config and merge it over the defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    if not isinstance(user_config, dict):
        raise TypeError(f"Config must be a JSON object: {path}")
    config.update(user_config)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required_positive = (
        "img_size",
        "patch_size",
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "batch_size",
        "tuned_epoch",
        "pool_size",
        "length",
        "top_k",
        "init_cls",
        "increment",
    )
    for key in required_positive:
        value = config.get(key)
        if value is None or float(value) <= 0:
            raise ValueError(f"{key} must be positive, got {value!r}")
    if config["img_size"] % config["patch_size"] != 0:
        raise ValueError("img_size must be divisible by patch_size")
    if config["embed_dim"] % config["num_heads"] != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if config["depth"] < 3:
        raise ValueError("depth must be at least 3 to retain a middle layer")
    if config["top_k"] > config["pool_size"]:
        raise ValueError("top_k cannot exceed pool_size")
    if config.get("drop_path_rate", 0.0) not in (0, 0.0):
        raise ValueError(
            "This port currently supports drop_path_rate=0, matching the "
            "released LDEPrompt configs. Set it to 0."
        )
    if config.get("importance_normalization") not in {"zscore", "softmax"}:
        raise ValueError("importance_normalization must be zscore or softmax")
    if config.get("optimizer") not in {"adam", "adamw", "sgd"}:
        raise ValueError("optimizer must be adam, adamw, or sgd")
    if config.get("scheduler") not in {"cosine", "constant", "steplr"}:
        raise ValueError("scheduler must be cosine, constant, or steplr")
    if config.get("scheduler") == "steplr":
        milestones = config.get("init_milestones", [])
        if not isinstance(milestones, Iterable) or isinstance(milestones, (str, bytes)):
            raise ValueError("init_milestones must be a list for the steplr scheduler")
        if not list(milestones):
            raise ValueError("init_milestones cannot be empty for the steplr scheduler")
        if float(config.get("init_lr_decay", 0.0)) <= 0:
            raise ValueError("init_lr_decay must be positive for the steplr scheduler")
    if int(config.get("num_workers", 0)) < 0:
        raise ValueError("num_workers cannot be negative")
    if int(config.get("ca_batch_size", 256)) <= 0:
        raise ValueError("ca_batch_size must be positive")
    if config.get("ca_storage_efficient_method") not in {
        "covariance",
        "variance",
        "multi-centroid",
    }:
        raise ValueError("Unsupported classifier-alignment statistics method")


def resolve_seeds(config: Dict[str, Any], override: int | None = None) -> List[int]:
    if override is not None:
        return [int(override)]
    seed_value = config.get("seed", 1993)
    if isinstance(seed_value, Iterable) and not isinstance(seed_value, (str, bytes)):
        seeds = [int(seed) for seed in seed_value]
    else:
        seeds = [int(seed_value)]
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return seeds
