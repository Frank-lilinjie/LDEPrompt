from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ldeprompt_mindspore.model import LDEPromptViT
from scripts.convert_jax_vit import convert_arrays, convert_file
from tests.support import tiny_config


def _fake_jax_weights(dim=8, depth=3, heads=2, mlp=16, patches=5):
    head_dim = dim // heads
    arrays = {
        "embedding/kernel": np.random.RandomState(1).normal(size=(4, 4, 3, dim)),
        "embedding/bias": np.zeros(dim),
        "cls": np.zeros((1, 1, dim)),
        "Transformer/posembed_input/pos_embedding": np.zeros((1, patches, dim)),
        "Transformer/encoder_norm/scale": np.ones(dim),
        "Transformer/encoder_norm/bias": np.zeros(dim),
    }
    for index in range(depth):
        prefix = f"Transformer/encoderblock_{index}/"
        attention = prefix + "MultiHeadDotProductAttention_1/"
        arrays[prefix + "LayerNorm_0/scale"] = np.ones(dim)
        arrays[prefix + "LayerNorm_0/bias"] = np.zeros(dim)
        for component in ("query", "key", "value"):
            arrays[attention + component + "/kernel"] = np.random.RandomState(index + 2).normal(
                size=(dim, heads, head_dim)
            )
            arrays[attention + component + "/bias"] = np.zeros((heads, head_dim))
        arrays[attention + "out/kernel"] = np.random.RandomState(index + 7).normal(
            size=(heads, head_dim, dim)
        )
        arrays[attention + "out/bias"] = np.zeros(dim)
        arrays[prefix + "MlpBlock_3/Dense_0/kernel"] = np.random.RandomState(index + 11).normal(
            size=(dim, mlp)
        )
        arrays[prefix + "MlpBlock_3/Dense_0/bias"] = np.zeros(mlp)
        arrays[prefix + "MlpBlock_3/Dense_1/kernel"] = np.random.RandomState(index + 13).normal(
            size=(mlp, dim)
        )
        arrays[prefix + "MlpBlock_3/Dense_1/bias"] = np.zeros(dim)
        arrays[prefix + "LayerNorm_2/scale"] = np.ones(dim)
        arrays[prefix + "LayerNorm_2/bias"] = np.zeros(dim)
    return arrays


class WeightConversionTests(unittest.TestCase):
    def test_conversion_shapes(self):
        converted = convert_arrays(_fake_jax_weights(), verbose=False)
        self.assertEqual(converted["patch_embed.proj.weight"].shape, (8, 3, 4, 4))
        self.assertEqual(converted["blocks.0.attn.qkv.weight"].shape, (24, 8))
        self.assertEqual(converted["blocks.0.attn.proj.weight"].shape, (8, 8))
        self.assertEqual(converted["blocks.0.mlp.fc1.weight"].shape, (16, 8))

    def test_converted_checkpoint_loads_into_backbone(self):
        config = tiny_config(mlp_ratio=2.0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "vit.npz"
            destination = Path(directory) / "vit.ckpt"
            np.savez(source, **_fake_jax_weights(patches=17))
            count = convert_file(source, destination)
            model = LDEPromptViT(config, num_classes=6)
            info = model.load_backbone_checkpoint(destination)

        self.assertGreater(count, 40)
        self.assertEqual(info["skipped"], 0)
        self.assertFalse(model.patch_embed.proj.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
