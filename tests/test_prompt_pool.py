from __future__ import annotations

import unittest

import numpy as np

import mindspore as ms

from ldeprompt_mindspore.prompt import PromptCollection, select_reusable_prompt_ids


class PromptReuseRuleTests(unittest.TestCase):
    def test_threshold_and_limit(self):
        scores = np.asarray([0.1, 0.2, 0.8, 1.0], dtype=np.float32)
        self.assertEqual(select_reusable_prompt_ids(scores, max_old=1), [3])
        self.assertEqual(select_reusable_prompt_ids(scores, max_old=3), [2, 3])

    def test_no_positive_similarity_means_no_reuse(self):
        self.assertEqual(select_reusable_prompt_ids(np.asarray([-0.2, -0.1]), 2), [])
        self.assertEqual(select_reusable_prompt_ids(np.asarray([0.1, 0.2]), 0), [])


class PromptPoolTests(unittest.TestCase):
    def _collection(self):
        return PromptCollection(
            depth=3,
            length=1,
            embed_dim=8,
            num_heads=2,
            top_k=1,
            pool_size=2,
            batchwise_prompt=False,
            prompt_init="uniform",
            prompt_key_init="uniform",
            same_key_value=False,
            num_placeholder=1,
        )

    def test_freeze_expand_and_merge(self):
        pool = self._collection().layer(1)
        pool.prepare_training([], add_num=2)
        self.assertEqual(len(pool.train_prompts), 2)
        self.assertTrue(pool.train_prompts[0].prompt.requires_grad)
        pool.merge_new_into_global()
        self.assertEqual(len(pool.global_prompts), 2)
        self.assertFalse(pool.train_prompts[0].prompt.requires_grad)

        pool.prepare_training([1], add_num=1)
        self.assertEqual(len(pool.train_prompts), 2)
        self.assertFalse(pool.train_prompts[0].prompt.requires_grad)
        self.assertTrue(pool.train_prompts[1].prompt.requires_grad)
        new_value = pool.train_prompts[1].prompt.asnumpy().copy()
        pool.merge_new_into_global()
        self.assertEqual(len(pool.global_prompts), 3)
        np.testing.assert_allclose(pool.global_prompts[-1].prompt.asnumpy(), new_value)

    def test_placeholder_routing(self):
        pool = self._collection().layer(1)
        pool.prepare_training([], add_num=1)
        pool.merge_new_into_global()
        dim = pool.embed_dim

        unit0 = np.zeros((1, dim), np.float32)
        unit1 = np.zeros((1, dim), np.float32)
        unit0[0, 0] = 1.0
        unit1[0, 1] = 1.0
        features0 = np.broadcast_to(unit0[:, None, :], (2, 5, dim)).copy()
        features1 = np.broadcast_to(unit1[:, None, :], (2, 5, dim)).copy()

        pool.global_prompts[0].prompt_key.set_data(ms.Tensor(unit0))
        pool.placeholders[0].placeholder.set_data(ms.Tensor(-unit0))
        selected = pool.select_eval_prompts(ms.Tensor(features0))
        self.assertEqual(selected["need_prompt_mask"].tolist(), [True, True])

        pool.placeholders[0].placeholder.set_data(ms.Tensor(unit1))
        selected = pool.select_eval_prompts(ms.Tensor(features1))
        self.assertEqual(selected["need_prompt_mask"].tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
