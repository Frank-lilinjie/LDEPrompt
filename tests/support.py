from __future__ import annotations

from copy import deepcopy

from ldeprompt_mindspore.config import DEFAULT_CONFIG, validate_config


def tiny_config(**overrides):
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "dataset": "synthetic",
            "synthetic_classes": 6,
            "synthetic_samples_per_class": 6,
            "synthetic_eval_samples_per_class": 3,
            "synthetic_noise": 0.03,
            "synthetic_seed": 123,
            "max_samples_per_class": 3,
            "eval_samples_per_class": 2,
            "shuffle_classes": False,
            "init_cls": 3,
            "increment": 3,
            "max_tasks": 2,
            "img_size": 16,
            "patch_size": 4,
            "embed_dim": 8,
            "depth": 3,
            "num_heads": 2,
            "batch_size": 3,
            "tuned_epoch": 1,
            "pool_size": 2,
            "length": 1,
            "top_k": 1,
            "num_placeholder": 1,
            "importance_max_batches": 1,
            "crct_epochs": 0,
            "ca_batch_size": 8,
            "train_augment": False,
            "eval_resize": False,
            "num_workers": 0,
            "save_checkpoint": False,
            "seed": 42,
        }
    )
    config.update(overrides)
    validate_config(config)
    return config
