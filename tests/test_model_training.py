from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import mindspore as ms

from ldeprompt_mindspore.data import ContinualDataManager
from ldeprompt_mindspore.model import LDEPromptViT
from ldeprompt_mindspore.trainer import LDEPromptLearner, build_logit_mask
from tests.support import tiny_config


def _prepare_model(config, num_classes=6):
    model = LDEPromptViT(config, num_classes)
    model.class_to_task = [0, 0, 0, 1, 1, 1]
    model.prepare_task_prompts(0)
    model.prompt_pool.prepare_task([1], {1: []}, {1: 2})
    model.merge_task_prompts()
    model.prepare_task_prompts(1)
    model.prompt_pool.prepare_task([1], {1: [0]}, {1: 1})
    model.merge_task_prompts()
    prototypes = np.random.RandomState(5).normal(size=(num_classes, config["embed_dim"])).astype(np.float32)
    model.task_indicator.set_data(ms.Tensor(prototypes))
    model.head.weight.set_data(
        ms.Tensor(np.random.RandomState(6).normal(size=(num_classes, config["embed_dim"])).astype(np.float32))
    )
    model.head.bias.set_data(ms.Tensor(np.arange(num_classes, dtype=np.float32)))
    return model


class TrainingMaskTests(unittest.TestCase):
    def test_old_and_future_classes_are_masked(self):
        mask = build_logit_mask(6, known_classes=2, total_classes=4)
        self.assertTrue(np.all(mask[:2] == -1.0e4))
        self.assertTrue(np.all(mask[2:4] == 0.0))
        self.assertTrue(np.all(mask[4:] == -1.0e4))


class RoutingTests(unittest.TestCase):
    def test_seen_class_mask_blocks_future_task(self):
        config = tiny_config()
        model = LDEPromptViT(config, num_classes=2)
        prototypes = np.asarray([[0.8, 0.6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], np.float32)
        model.task_indicator.set_data(ms.Tensor(prototypes))
        model.class_to_task = [0, 1]
        features = ms.Tensor(np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))

        self.assertEqual(model.route_task_ids(features).tolist(), [1])
        self.assertEqual(model.route_task_ids(features, num_classes=1).tolist(), [0])


class CheckpointTests(unittest.TestCase):
    def test_bundle_round_trip(self):
        config = tiny_config()
        model = _prepare_model(config)
        images = ms.Tensor(
            np.random.RandomState(11).uniform(size=(2, 3, 16, 16)).astype(np.float32)
        )
        expected = model.forward_eval(images, task_ids=[0, 5])["logits"].asnumpy()

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "model"
            model.save_bundle(prefix)
            saved = ms.load_checkpoint(str(prefix.with_suffix(".ckpt")))
            self.assertFalse(any("train_prompts" in name for name in saved))

            restored = LDEPromptViT(config, num_classes=6)
            result = restored.restore_bundle(prefix)
            actual = restored.forward_eval(images, task_ids=[0, 5])["logits"].asnumpy()
            metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertEqual(result["result"], ([], []))
        self.assertEqual(metadata["pools"]["1"]["global_prompts"], 3)
        self.assertEqual(restored.prompt_pool.selected_layers, [1])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class EndToEndTests(unittest.TestCase):
    def test_two_task_training_and_evaluation(self):
        config = tiny_config()
        manager = ContinualDataManager(config)
        with tempfile.TemporaryDirectory() as directory:
            learner = LDEPromptLearner(config, manager, output_dir=directory)
            metrics = learner.run()
            self.assertTrue((Path(directory) / "metrics.json").is_file())

        self.assertEqual(metrics["num_tasks"], 2)
        self.assertEqual(len(metrics["top1_curve"]), 2)
        self.assertEqual(metrics["task_classes"], [[0, 1, 2], [3, 4, 5]])
        self.assertFalse(metrics["paper_reproduced"])


if __name__ == "__main__":
    unittest.main()
