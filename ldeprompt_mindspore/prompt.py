"""Prefix prompts and expandable prompt pools for LDEPrompt on MindSpore."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from mindspore import nn


def _prompt_array(
    length: int,
    embed_dim: int,
    num_heads: int,
    prompt_init: str,
    same_key_value: bool,
    seed: int,
) -> np.ndarray:
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    rng = np.random.RandomState(seed)
    head_dim = embed_dim // num_heads
    if prompt_init == "zero":
        prompt = np.zeros((1, 2, length, num_heads, head_dim), dtype=np.float32)
    elif prompt_init == "uniform":
        prompt = rng.uniform(-1.0, 1.0, size=(1, 2, length, num_heads, head_dim)).astype(np.float32)
    else:
        raise ValueError("prompt_init must be uniform or zero")
    if same_key_value:
        prompt[:, 1] = prompt[:, 0]
    return prompt


def _key_array(embed_dim: int, key_init: str, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if key_init == "zero":
        return np.zeros((1, embed_dim), dtype=np.float32)
    if key_init == "uniform":
        return rng.uniform(-1.0, 1.0, size=(1, embed_dim)).astype(np.float32)
    raise ValueError("prompt_key_init must be uniform or zero")


def select_reusable_prompt_ids(mean_similarity: np.ndarray, max_old: int) -> List[int]:
    """Select historically reusable prompts using the released-code rule.

    The repository keeps prompt keys whose mean similarity is positive and above
    the mean of all positive prompt similarities. At most ``max_old`` keys are
    reused so that every task can still add at least one new prompt.
    """
    scores = np.asarray(mean_similarity, dtype=np.float64).reshape(-1)
    limit = max(0, int(max_old))
    if scores.size == 0 or limit == 0:
        return []
    positive = scores[scores > 0]
    if positive.size == 0:
        return []
    threshold = float(positive.mean())
    candidates = np.flatnonzero(scores > threshold)
    if len(candidates) > limit:
        order = np.argsort(-scores[candidates], kind="stable")[:limit]
        candidates = candidates[order]
    return [int(index) for index in candidates]


def make_prompt_class():
    """Import MindSpore lazily so config/data utilities remain importable."""
    import mindspore as ms
    from mindspore import nn

    class Prompt(nn.Cell):
        """A trainable prefix prompt with its routing key."""

        def __init__(
            self,
            length: int,
            embed_dim: int,
            num_heads: int,
            prompt_init: str = "uniform",
            prompt_key_init: str = "uniform",
            same_key_value: bool = False,
            use_prompt_key: bool = True,
            seed: int = 0,
        ):
            super().__init__()
            self.length = int(length)
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.same_key_value = bool(same_key_value)
            self.use_prompt_key = bool(use_prompt_key)
            prompt = _prompt_array(
                length, embed_dim, num_heads, prompt_init, same_key_value, seed
            )
            self.prompt = ms.Parameter(ms.Tensor(prompt), name=f"prompt_{seed}")
            if use_prompt_key:
                key = _key_array(embed_dim, prompt_key_init, seed + 1)
                self.prompt_key = ms.Parameter(ms.Tensor(key), name=f"prompt_key_{seed}")
            else:
                self.prompt_key = ms.Parameter(
                    ms.Tensor(np.zeros((1, embed_dim), dtype=np.float32)),
                    name=f"prompt_key_{seed}",
                    requires_grad=False,
                )

        def forward_features(self):
            return self.prompt[0, 0].reshape(-1)

        @classmethod
        def from_arrays(
            cls,
            prompt: np.ndarray,
            prompt_key: np.ndarray,
            requires_grad: bool,
        ) -> "Prompt":
            instance = cls.__new__(cls)
            nn.Cell.__init__(instance)
            instance.length = int(prompt.shape[2])
            instance.embed_dim = int(prompt.shape[-1] * prompt.shape[-2])
            instance.num_heads = int(prompt.shape[-2])
            instance.same_key_value = bool(np.array_equal(prompt[:, 0], prompt[:, 1]))
            instance.use_prompt_key = True
            instance.prompt = ms.Parameter(ms.Tensor(prompt.copy()), requires_grad=requires_grad)
            instance.prompt_key = ms.Parameter(
                ms.Tensor(prompt_key.copy()), requires_grad=requires_grad
            )
            return instance

        def clone(self, requires_grad: bool = False) -> "Prompt":
            return Prompt.from_arrays(
                self.prompt.asnumpy().copy(),
                self.prompt_key.asnumpy().copy(),
                requires_grad=requires_grad,
            )

        def set_trainable(self, trainable: bool) -> None:
            self.prompt.requires_grad = bool(trainable)
            self.prompt_key.requires_grad = bool(trainable)

    return Prompt


class LayerPromptPool(nn.Cell):
    """Global evaluate pool and per-task training pool for one ViT layer."""

    def __init__(
        self,
        depth_index: int,
        length: int,
        embed_dim: int,
        num_heads: int,
        top_k: int,
        pool_size: int,
        batchwise_prompt: bool,
        prompt_init: str,
        prompt_key_init: str,
        same_key_value: bool,
        num_placeholder: int,
    ):
        import mindspore as ms
        from mindspore import ops

        Prompt = make_prompt_class()
        super().__init__()
        self._ms = ms
        self._ops = ops
        self.Prompt = Prompt
        self.depth_index = int(depth_index)
        self.length = int(length)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.batchwise_prompt = bool(batchwise_prompt)
        self.prompt_init = str(prompt_init)
        self.prompt_key_init = str(prompt_key_init)
        self.same_key_value = bool(same_key_value)
        self.num_placeholder = int(num_placeholder)
        self.global_prompts = nn.CellList()
        self.train_prompts = nn.CellList()
        self.placeholders = nn.CellList()
        self.reused_prompt_ids: List[int] = []
        self.new_prompt_count = 0
        self.is_training = False

    def _new_prompt(self, seed: int, trainable: bool = True):
        prompt = self.Prompt(
            length=self.length,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            prompt_init=self.prompt_init,
            prompt_key_init=self.prompt_key_init,
            same_key_value=self.same_key_value,
            use_prompt_key=True,
            seed=seed,
        )
        prompt.set_trainable(trainable)
        return prompt

    def _add_placeholder(self, seed: int, trainable: bool = True):
        parameter_factory = self._ms.Parameter
        tensor_factory = self._ms.Tensor

        class LocalPlaceholder(nn.Cell):
            def __init__(self, dim: int, local_seed: int):
                super().__init__()
                local_rng = np.random.RandomState(local_seed)
                value = local_rng.uniform(-1.0, 1.0, size=(1, dim)).astype(np.float32)
                self.placeholder = parameter_factory(tensor_factory(value))

            def set_trainable(self, trainable: bool) -> None:
                self.placeholder.requires_grad = bool(trainable)

        placeholder = LocalPlaceholder(self.embed_dim, seed)
        placeholder.set_trainable(trainable)
        return placeholder

    def prepare_training(self, old_prompt_ids: Sequence[int], add_num: int) -> None:
        """Build the current task pool, freezing all reused prompts."""
        self.reused_prompt_ids = [int(index) for index in old_prompt_ids]
        self.new_prompt_count = int(add_num)
        self.train_prompts = nn.CellList()
        for index in self.reused_prompt_ids:
            if index < 0 or index >= len(self.global_prompts):
                raise IndexError(f"Invalid old prompt id {index} at layer {self.depth_index}")
            self.train_prompts.append(self.global_prompts[index].clone(requires_grad=False))
        base_seed = 1009 * (self.depth_index + 1) + 97 * len(self.global_prompts)
        for offset in range(self.new_prompt_count):
            self.train_prompts.append(self._new_prompt(base_seed + offset, trainable=True))
        self.placeholders = nn.CellList()
        for offset in range(self.num_placeholder):
            self.placeholders.append(self._add_placeholder(base_seed + 7919 + offset, True))
        self.is_training = True

    def freeze_training_pool(self) -> None:
        for prompt in self.train_prompts:
            prompt.set_trainable(False)
        for placeholder in self.placeholders:
            placeholder.set_trainable(False)

    def merge_new_into_global(self) -> int:
        """Append only the newly initialized prompts to the frozen global pool."""
        if self.new_prompt_count <= 0:
            self.freeze_training_pool()
            self.is_training = False
            return 0
        for prompt in list(self.train_prompts)[-self.new_prompt_count :]:
            self.global_prompts.append(prompt.clone(requires_grad=False))
        self.freeze_training_pool()
        self.is_training = False
        return self.new_prompt_count

    def restore_global_pool(self, count: int, placeholder_count: int) -> None:
        self.global_prompts = nn.CellList()
        base_seed = 3001 * (self.depth_index + 1)
        for index in range(int(count)):
            self.global_prompts.append(self._new_prompt(base_seed + index, trainable=False))
        self.placeholders = nn.CellList()
        for index in range(int(placeholder_count)):
            self.placeholders.append(self._add_placeholder(base_seed + 5000 + index, False))
        self.train_prompts = nn.CellList()
        self.is_training = False
        self.new_prompt_count = 0
        self.reused_prompt_ids = []

    def has_global_prompts(self) -> bool:
        return len(self.global_prompts) > 0

    def _embedding(self, features, cls_features=None):
        if cls_features is not None:
            return cls_features
        return features.max(axis=1)

    @staticmethod
    def _normalize(value):
        from mindspore import ops

        return ops.L2Normalize(axis=1, epsilon=1e-6)(value)

    def _topk_indices(self, similarity, prompt_count: int) -> np.ndarray:
        if prompt_count <= 0:
            raise RuntimeError("Prompt pool is empty")
        k = min(self.top_k, prompt_count)
        topk = self._ops.TopK(sorted=True)(similarity, k)
        indices = topk[1].asnumpy().astype(np.int64)
        if not self.batchwise_prompt:
            return indices
        flat = indices.reshape(-1)
        counts = np.bincount(flat, minlength=prompt_count)
        order = np.argsort(-counts, kind="stable")[:k]
        return np.repeat(order.reshape(1, -1), indices.shape[0], axis=0)

    def _assemble(self, prompts: Sequence, indices: np.ndarray):
        ops = self._ops
        rows, k = indices.shape
        selected = []
        for row in range(rows):
            for rank in range(k):
                selected.append(prompts[int(indices[row, rank])].prompt)
        stacked = ops.stack(selected, axis=0)  # [B*K, 1, 2, L, H, D]
        stacked = stacked.reshape(rows, k, *stacked.shape[2:])  # [B, K, 2, L, H, D]
        stacked = stacked.transpose(0, 2, 1, 3, 4, 5)  # [B, 2, K, L, H, D]
        return stacked.reshape(rows, 2, k * self.length, self.num_heads, self.embed_dim // self.num_heads)

    def select_train_prompts(self, features, cls_features=None) -> Dict[str, object]:
        if not self.train_prompts:
            raise RuntimeError(f"No training prompts at layer {self.depth_index}")
        ops = self._ops
        prompt_keys = ops.cat([prompt.prompt_key for prompt in self.train_prompts], axis=0)
        feature = self._embedding(features, cls_features)
        similarity = ops.matmul(self._normalize(feature), self._normalize(prompt_keys).transpose())
        indices = self._topk_indices(similarity, len(self.train_prompts))
        return {
            "prompt": self._assemble(self.train_prompts, indices),
            "indices": indices,
            "similarity": similarity,
        }

    def select_eval_prompts(self, features, cls_features=None) -> Dict[str, object]:
        if not self.global_prompts:
            raise RuntimeError(f"No global prompts at layer {self.depth_index}")
        ops = self._ops
        prompt_keys = ops.cat([prompt.prompt_key for prompt in self.global_prompts], axis=0)
        placeholder_keys = (
            ops.cat([placeholder.placeholder for placeholder in self.placeholders], axis=0)
            if len(self.placeholders) > 0
            else None
        )
        feature = self._embedding(features, cls_features)
        all_keys = prompt_keys if placeholder_keys is None else ops.cat([prompt_keys, placeholder_keys], axis=0)
        similarity = ops.matmul(self._normalize(feature), self._normalize(all_keys).transpose())
        top1 = self._ops.TopK(sorted=True)(similarity, 1)[1].asnumpy().astype(np.int64).reshape(-1)
        need_prompt_mask = top1 < len(self.global_prompts)
        prompt_similarity = similarity[:, : len(self.global_prompts)]
        indices = self._topk_indices(prompt_similarity, len(self.global_prompts))
        prompt = self._assemble(self.global_prompts, indices)
        if not np.all(need_prompt_mask):
            mask = self._ms.Tensor(need_prompt_mask.astype(np.float32)).reshape(-1, 1, 1, 1, 1)
            prompt = prompt * mask
        return {
            "prompt": prompt,
            "indices": indices,
            "similarity": similarity,
            "need_prompt_mask": need_prompt_mask,
        }


class PromptCollection(nn.Cell):
    """All layer-wise prompt banks used by one ViT."""

    def __init__(
        self,
        depth: int,
        length: int,
        embed_dim: int,
        num_heads: int,
        top_k: int,
        pool_size: int,
        batchwise_prompt: bool,
        prompt_init: str,
        prompt_key_init: str,
        same_key_value: bool,
        num_placeholder: int,
    ):
        super().__init__()
        self.layers = nn.CellList(
            [
                LayerPromptPool(
                    depth_index=index,
                    length=length,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    top_k=top_k,
                    pool_size=pool_size,
                    batchwise_prompt=batchwise_prompt,
                    prompt_init=prompt_init,
                    prompt_key_init=prompt_key_init,
                    same_key_value=same_key_value,
                    num_placeholder=num_placeholder,
                )
                for index in range(depth)
            ]
        )
        self.selected_layers: List[int] = []

    def layer(self, index: int) -> LayerPromptPool:
        return self.layers[index]

    def prepare_task(
        self,
        selected_layers: Sequence[int],
        reused_prompt_ids: Dict[int, Sequence[int]],
        add_counts: Dict[int, int],
    ) -> None:
        self.selected_layers = [int(layer) for layer in selected_layers]
        for layer in self.selected_layers:
            self.layer(layer).prepare_training(
                reused_prompt_ids.get(layer, ()), add_counts[layer]
            )

    def merge_new_prompts(self) -> Dict[int, int]:
        merged: Dict[int, int] = {}
        for layer in self.selected_layers:
            merged[layer] = self.layer(layer).merge_new_into_global()
        return merged

    def has_global_pool(self, layer: int) -> bool:
        return self.layer(layer).has_global_prompts()

    def restore_counts(self, global_counts: Dict[int, int], placeholder_counts: Dict[int, int]) -> None:
        for layer in range(len(self.layers)):
            self.layer(layer).restore_global_pool(
                global_counts.get(layer, 0), placeholder_counts.get(layer, 0)
            )
        self.selected_layers = sorted(
            int(layer) for layer, count in global_counts.items() if int(count) > 0
        )

    def pool_snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            str(layer): {
                "global_prompts": len(self.layer(layer).global_prompts),
                "placeholders": len(self.layer(layer).placeholders),
            }
            for layer in range(len(self.layers))
        }
