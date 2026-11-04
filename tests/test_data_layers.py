from __future__ import annotations

import unittest

import numpy as np

from ldeprompt_mindspore.data import (
    ContinualDataManager,
    _make_synthetic_split,
    _make_synthetic_templates,
)
from ldeprompt_mindspore.model import select_important_layers
from tests.support import tiny_config


class SyntheticDataTests(unittest.TestCase):
    def test_train_and_test_share_class_templates(self):
        templates = _make_synthetic_templates(4, 16, 77)
        train_x, train_y = _make_synthetic_split(4, 16, 16, 0.03, 100, templates)
        test_x, test_y = _make_synthetic_split(4, 16, 16, 0.03, 900, templates)

        train_means = np.stack([train_x[train_y == index].mean(axis=0) for index in range(4)])
        test_means = np.stack([test_x[test_y == index].mean(axis=0) for index in range(4)])
        within_class = np.abs(train_means - test_means).mean()
        expected = templates.transpose(0, 2, 3, 1) * 255.0
        expected_error = np.abs(train_means - expected).mean() / 255.0
        cross_class = np.abs(train_means[:, None] - train_means[None, :]).mean()

        self.assertLess(expected_error, 0.01)
        self.assertLess(within_class, cross_class * 0.1)

    def test_task_layout_uses_only_requested_tasks(self):
        manager = ContinualDataManager(tiny_config())
        self.assertEqual(manager.task_classes, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(manager.increments, [3, 3])
        self.assertEqual(len(manager.indices_for("train", [4])), 3)
        self.assertEqual(len(manager.indices_for("test", [1])), 2)

    def test_parallel_loader_is_serializable(self):
        manager = ContinualDataManager(tiny_config())
        loader = manager.train_loader([0, 1], batch_size=2, shuffle=False, num_workers=2)
        totals = 0
        for images, labels in loader.create_tuple_iterator(num_epochs=1):
            self.assertEqual(tuple(images.shape[1:]), (3, 16, 16))
            totals += len(labels)
        self.assertEqual(totals, 6)


class LayerSelectionTests(unittest.TestCase):
    def test_middle_layers_only(self):
        gains = [10.0, 1.0, 5.0, 2.0, 9.0]
        selected = select_important_layers(gains, normalization="zscore", depth=5)
        self.assertIn(2, selected)
        self.assertNotIn(0, selected)
        self.assertNotIn(4, selected)

        softmax_selected = select_important_layers(
            gains, normalization="softmax", depth=5
        )
        self.assertNotIn(0, softmax_selected)
        self.assertNotIn(4, softmax_selected)

    def test_empty_selection_falls_back_to_best_middle_layer(self):
        self.assertEqual(select_important_layers([0, 0, 0, 0], depth=4), [1])

    def test_invalid_layer_count_is_rejected(self):
        with self.assertRaises(ValueError):
            select_important_layers([1.0, 2.0], depth=2)


if __name__ == "__main__":
    unittest.main()
