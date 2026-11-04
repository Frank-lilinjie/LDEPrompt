"""MindSpore ViT-B/16 with layer-selective prefix prompts for LDEPrompt."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from .prompt import PromptCollection

class PatchEmbed(nn.Cell):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            has_bias=True,
            pad_mode="valid",
        )

    def construct(self, x):
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x.transpose(0, 2, 1)


class PrefixAttention(nn.Cell):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = float(self.head_dim) ** -0.5
        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.proj = nn.Dense(dim, dim, has_bias=True)

    def construct(self, x, prompt=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        if prompt is not None:
            prefix = prompt.transpose(1, 0, 3, 2, 4)
            key = ops.cat((prefix[0], key), axis=2)
            value = ops.cat((prefix[1], value), axis=2)
        attention = ops.matmul(query, key.transpose(0, 1, 3, 2)) * self.scale
        attention = ops.softmax(attention, axis=-1)
        output = ops.matmul(attention, value)
        output = output.transpose(0, 2, 1, 3).reshape(batch, tokens, channels)
        return self.proj(output)


class MLP(nn.Cell):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Dense(dim, hidden_dim, has_bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Dense(hidden_dim, dim, has_bias=True)

    def construct(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Cell):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.attn = PrefixAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def construct(self, x, prompt=None):
        x = x + self.attn(self.norm1(x), prompt)
        x = x + self.mlp(self.norm2(x))
        return x


def entropy_per_token(tensor, eps: float = 1e-8):
    probabilities = ops.softmax(tensor, axis=-1)
    return -(probabilities * ops.log(probabilities + eps)).sum(axis=-1)


def select_important_layers(
    gains: Sequence[float], normalization: str = "zscore", depth: int | None = None
) -> List[int]:
    """Select middle layers with positive normalized information gain."""
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("gains must be a 1-D sequence with at least three layers")
    if depth is None:
        depth = values.size
    middle = np.arange(1, depth - 1)
    middle_values = values[middle]
    if normalization == "zscore":
        mean = float(middle_values.mean())
        std = float(middle_values.std())
        scores = (values - mean) / (std + 1e-6)
    elif normalization == "softmax":
        shifted = values - float(values.max())
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        scores = probabilities
    else:
        raise ValueError("importance_normalization must be zscore or softmax")
    threshold = 0.0 if normalization == "zscore" else float(scores[middle].mean())
    selected = [int(index) for index in middle if scores[index] > threshold]
    if not selected:
        selected = [int(middle[int(np.argmax(middle_values))])]
    return selected


class LDEPromptViT(nn.Cell):
    """ViT with g/task prefixes, selective layer prompts, and a fixed class head."""

    def __init__(self, config: Dict[str, Any], num_classes: int):
        super().__init__()
        self.config = dict(config)
        self.img_size = int(config["img_size"])
        self.patch_size = int(config["patch_size"])
        self.in_chans = int(config.get("in_chans", 3))
        self.embed_dim = int(config["embed_dim"])
        self.depth = int(config["depth"])
        self.num_heads = int(config["num_heads"])
        self.mlp_ratio = float(config.get("mlp_ratio", 4.0))
        self.qkv_bias = bool(config.get("qkv_bias", True))
        self.prompt_length = int(config["length"])
        self.num_classes = int(num_classes)

        self.patch_embed = PatchEmbed(
            self.img_size, self.patch_size, self.in_chans, self.embed_dim
        )
        self.cls_token = ms.Parameter(
            ms.Tensor(np.zeros((1, 1, self.embed_dim), dtype=np.float32)), name="cls_token"
        )
        self.pos_embed = ms.Parameter(
            ms.Tensor(
                np.random.RandomState(int(config.get("seed", 1993)))
                .normal(0.0, 0.02, size=(1, self.patch_embed.num_patches + 1, self.embed_dim))
                .astype(np.float32)
            ),
            name="pos_embed",
        )
        self.blocks = nn.CellList(
            [
                Block(self.embed_dim, self.num_heads, self.mlp_ratio, self.qkv_bias)
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm((self.embed_dim,), epsilon=1e-6)
        self.head = nn.Dense(self.embed_dim, self.num_classes, has_bias=True)

        prompt_shape = (1, 2, self.prompt_length, self.num_heads, self.embed_dim // self.num_heads)
        prompt_rng = np.random.RandomState(int(config.get("seed", 1993)) + 11)
        if config.get("prompt_init", "uniform") == "zero":
            g_prompt = np.zeros(prompt_shape, dtype=np.float32)
        elif config.get("prompt_init", "uniform") == "uniform":
            g_prompt = prompt_rng.uniform(-1.0, 1.0, size=prompt_shape).astype(np.float32)
        else:
            raise ValueError("prompt_init must be uniform or zero")
        self.g_prompt = ms.Parameter(ms.Tensor(g_prompt), name="g_prompt")
        self.task_prompts = nn.CellList()
        self.prompt_pool = PromptCollection(
            depth=self.depth,
            length=self.prompt_length,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            top_k=int(config["top_k"]),
            pool_size=int(config["pool_size"]),
            batchwise_prompt=bool(config.get("batchwise_prompt", True)),
            prompt_init=str(config.get("prompt_init", "uniform")),
            prompt_key_init=str(config.get("prompt_key_init", "uniform")),
            same_key_value=bool(config.get("same_key_value", False)),
            num_placeholder=int(config.get("num_placeholder", 0)),
        )
        self.task_indicator = ms.Parameter(
            ms.Tensor(np.zeros((self.num_classes, self.embed_dim), dtype=np.float32)),
            name="task_indicator",
            requires_grad=False,
        )
        self.class_to_task: List[int] = [-1] * self.num_classes
        self.current_task = -1
        self._add_task_prompt(0)
        self.freeze_backbone()

    @staticmethod
    def make_prompt_for_task(length: int, embed_dim: int, num_heads: int, init: str, seed: int):
        from .prompt import make_prompt_class

        Prompt = make_prompt_class()
        return Prompt(
            length=length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            prompt_init=init,
            prompt_key_init="zero",
            same_key_value=False,
            use_prompt_key=False,
            seed=seed,
        )

    def _add_task_prompt(self, task_id: int) -> None:
        while len(self.task_prompts) <= task_id:
            prompt = self.make_prompt_for_task(
                self.prompt_length,
                self.embed_dim,
                self.num_heads,
                str(self.config.get("prompt_init", "uniform")),
                int(self.config.get("seed", 1993)) + 997 * (len(self.task_prompts) + 1),
            )
            prompt.set_trainable(True)
            self.task_prompts.append(prompt)
        for index, prompt in enumerate(self.task_prompts):
            prompt.set_trainable(index == len(self.task_prompts) - 1)

    def freeze_backbone(self) -> None:
        names = ("patch_embed", "cls_token", "pos_embed", "blocks", "norm")
        for name, parameter in self.parameters_and_names():
            if name.startswith(names):
                parameter.requires_grad = False

    def trainable_task_parameters(self, task_id: int) -> List[ms.Parameter]:
        params: List[ms.Parameter] = [
            self.g_prompt,
            self.task_prompts[task_id].prompt,
            self.task_prompts[task_id].prompt_key,
        ]
        params.extend(self.prompt_pool.trainable_params())
        params.extend(self.head.trainable_params())
        return list({id(parameter): parameter for parameter in params}.values())

    def prepare_task_prompts(self, task_id: int) -> None:
        self.current_task = int(task_id)
        self._add_task_prompt(task_id)
        for index, prompt in enumerate(self.task_prompts):
            prompt.set_trainable(index == task_id)

    def _embed_tokens(self, x):
        x = self.patch_embed(x)
        cls_tokens = ops.broadcast_to(self.cls_token, (x.shape[0], 1, self.embed_dim))
        return ops.cat((cls_tokens, x), axis=1) + self.pos_embed

    def _task_prompt_batch(self, task_ids: Sequence[int], batch_size: int):
        prompt_list = []
        for task_id in task_ids:
            index = int(task_id)
            if index < 0 or index >= len(self.task_prompts):
                index = min(max(index, 0), len(self.task_prompts) - 1)
            prompt_list.append(self.task_prompts[index].prompt[0])
        if len(prompt_list) != batch_size:
            raise ValueError("task_ids length must match batch size")
        return ops.stack(prompt_list, axis=0)

    def _first_layer_prompt(self, task_ids: Sequence[int], batch_size: int):
        task_prompt = self._task_prompt_batch(task_ids, batch_size)
        g_prompt = ops.broadcast_to(self.g_prompt, task_prompt.shape)
        if bool(self.config.get("use_g_prompt", True)):
            return ops.cat((g_prompt, task_prompt), axis=2)
        return task_prompt

    def forward_features_train(self, x, task_id: int):
        task_ids = [int(task_id)] * x.shape[0]
        return self._forward_prompted(x, task_ids, training=True)

    def forward_features_eval(self, x, task_ids: Sequence[int]):
        return self._forward_prompted(x, task_ids, training=False)

    def _forward_prompted(self, x, task_ids: Sequence[int], training: bool):
        x = self._embed_tokens(x)
        first_prompt = self._first_layer_prompt(task_ids, x.shape[0])
        for index in range(self.depth):
            block = self.blocks[index]
            layer_pool = self.prompt_pool.layer(index)
            if index == 0:
                x = block(x, first_prompt)
            elif training and index in self.prompt_pool.selected_layers:
                selected = layer_pool.select_train_prompts(x)
                x = block(x, selected["prompt"])
            elif (not training) and layer_pool.has_global_prompts():
                selected = layer_pool.select_eval_prompts(x)
                x = block(x, selected["prompt"])
            else:
                x = block(x)
        return self.norm(x)

    def forward_train(self, x, task_id: int):
        tokens = self.forward_features_train(x, task_id)
        features = tokens[:, 0]
        return {"logits": self.head(features), "features": features, "tokens": tokens}

    def route_task_ids(self, features, num_classes: int | None = None) -> np.ndarray:
        normalized_features = ops.L2Normalize(axis=1, epsilon=1e-6)(features)
        normalized_prototypes = ops.L2Normalize(axis=1, epsilon=1e-6)(self.task_indicator)
        logits = ops.matmul(normalized_features, normalized_prototypes.transpose())
        if num_classes is not None and int(num_classes) < self.num_classes:
            visible = max(0, int(num_classes))
            mask = np.full((1, self.num_classes), -1.0e4, dtype=np.float32)
            mask[:, :visible] = 0.0
            logits = logits + ms.Tensor(mask, ms.float32)
        classes = ops.argmax(logits, 1).asnumpy().astype(np.int64)
        return np.asarray(
            [self.class_to_task[int(class_id)] if int(class_id) < len(self.class_to_task) else 0 for class_id in classes],
            dtype=np.int64,
        )

    def forward_eval(self, x, task_ids: Sequence[int] | None = None, num_classes: int | None = None):
        if task_ids is None:
            proto_features = self.no_prompt_features(x)
            task_ids = self.route_task_ids(proto_features, num_classes=num_classes)
        tokens = self.forward_features_eval(x, task_ids)
        features = tokens[:, 0]
        return {"logits": self.head(features), "features": features, "tokens": tokens}

    def construct(self, x, task_id: int = 0, training: bool = True):
        if training:
            return self.forward_train(x, task_id)
        return self.forward_eval(x, None)

    def no_prompt_features(self, x):
        x = self._embed_tokens(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]

    def features_after_layer(self, x, layer_index: int):
        x = self._embed_tokens(x)
        for index in range(int(layer_index) + 1):
            x = self.blocks[index](x)
        return self.norm(x)[:, 0]

    def features_at_layers(self, x, layer_indices: Iterable[int]) -> Dict[int, ms.Tensor]:
        requested = {int(index) for index in layer_indices}
        if not requested:
            return {}
        outputs: Dict[int, ms.Tensor] = {}
        x = self._embed_tokens(x)
        for index in range(self.depth):
            x = self.blocks[index](x)
            if index in requested:
                outputs[index] = self.norm(x)[:, 0]
        return outputs

    def information_gain_scores(self, loader, max_batches: int = 0) -> List[float]:
        """Estimate absolute entropy change for each encoder block on CPU/NPU."""
        gains = np.zeros(self.depth, dtype=np.float64)
        seen = 0
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            images = batch[0]
            x = self._embed_tokens(images)
            batch_size = int(images.shape[0])
            for index, block in enumerate(self.blocks):
                previous = x
                x = block(x)
                if 0 < index < self.depth - 1:
                    entropy_before = entropy_per_token(previous).mean(axis=1)
                    entropy_after = entropy_per_token(x).mean(axis=1)
                    gains[index] += float((entropy_after - entropy_before).abs().sum().asnumpy())
            seen += batch_size
        if seen == 0:
            raise RuntimeError("Information-gain loader yielded no samples")
        gains /= float(seen)
        logging.info("Information gain by block: %s", [round(float(value), 6) for value in gains])
        return gains.tolist()

    def update_task_prototypes(self, loader, task_id: int) -> None:
        """Store no-prompt CLS prototypes used by the task router."""
        sums: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        for batch in loader:
            images, labels = batch[0], batch[1].asnumpy()
            features = self.no_prompt_features(images).asnumpy()
            for row, label in enumerate(labels):
                class_id = int(label)
                sums[class_id] = sums.get(class_id, 0.0) + features[row]
                counts[class_id] = counts.get(class_id, 0) + 1
        prototypes = self.task_indicator.asnumpy()
        for class_id, total in sums.items():
            prototypes[class_id] = total / max(1, counts[class_id])
            self.class_to_task[class_id] = int(task_id)
        self.task_indicator.set_data(ms.Tensor(prototypes, ms.float32))
        logging.info("Updated %d task-routing prototypes for task %d", len(sums), task_id)

    def prompt_diversity_loss(self, selected_layers: Iterable[int], add_counts: Dict[int, int]):
        total = None
        pairs = 0
        for layer_index in selected_layers:
            add_count = int(add_counts[layer_index])
            if add_count <= 0:
                continue
            pool = self.prompt_pool.layer(layer_index)
            new_prompts = list(pool.train_prompts)[-add_count:]
            if len(pool.global_prompts) > add_count:
                old_prompts = list(pool.global_prompts)
            else:
                old_prompts = new_prompts
            new_features = ops.stack([prompt.forward_features().reshape((1, -1)) for prompt in new_prompts], axis=0)
            old_features = ops.stack([prompt.forward_features().reshape((1, -1)) for prompt in old_prompts], axis=0)
            new_features = ops.L2Normalize(axis=1, epsilon=1e-6)(new_features.squeeze(1))
            old_features = ops.L2Normalize(axis=1, epsilon=1e-6)(old_features.squeeze(1))
            similarity = ops.matmul(new_features, old_features.transpose())
            current = similarity.sum()
            total = current if total is None else total + current
            pairs += int(similarity.size)
        if total is None or pairs == 0:
            return ms.Tensor(0.0, ms.float32)
        return total / float(pairs)

    def placeholder_orthogonality_loss(self, selected_layers: Iterable[int]):
        total = None
        count = 0
        for layer_index in selected_layers:
            pool = self.prompt_pool.layer(layer_index)
            if len(pool.train_prompts) == 0 or len(pool.placeholders) == 0:
                continue
            keys = ops.cat([prompt.prompt_key for prompt in pool.train_prompts], axis=0)
            placeholders = ops.cat([item.placeholder for item in pool.placeholders], axis=0)
            keys = ops.L2Normalize(axis=1, epsilon=1e-6)(keys)
            placeholders = ops.L2Normalize(axis=1, epsilon=1e-6)(placeholders)
            cross = ops.matmul(placeholders, keys.transpose())
            layer_loss = (cross ** 2).mean()
            if placeholders.shape[0] > 1:
                between = ops.matmul(placeholders, placeholders.transpose())
                mask = ops.triu(ops.ones_like(between), diagonal=1)
                layer_loss = layer_loss + 0.1 * ((between * mask) ** 2).sum() / mask.sum()
            total = layer_loss if total is None else total + layer_loss
            count += 1
        if total is None or count == 0:
            return ms.Tensor(0.0, ms.float32)
        return total / float(count)

    def merge_task_prompts(self) -> None:
        self.prompt_pool.merge_new_prompts()
        for index, prompt in enumerate(self.task_prompts):
            prompt.set_trainable(False)

    def load_backbone_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Backbone checkpoint not found: {path}")
        parameters = ms.load_checkpoint(str(path))
        local = {name: parameter for name, parameter in self.parameters_and_names()}
        selected = {}
        skipped = 0
        for name, value in parameters.items():
            clean_name = name
            for prefix in ("backbone.", "model.", "network.backbone."):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
            if clean_name not in local or tuple(value.shape) != tuple(local[clean_name].shape):
                skipped += 1
                continue
            selected[clean_name] = value
        if not selected:
            raise ValueError(f"No compatible ViT parameters found in {path}")
        result = ms.load_param_into_net(self, selected, strict_load=False)
        logging.info(
            "Loaded %d backbone tensors; skipped %d incompatible tensors (%s)",
            len(selected),
            skipped,
            result,
        )
        self.freeze_backbone()
        return {"loaded": len(selected), "skipped": skipped, "path": str(path)}

    def state_metadata(self) -> Dict[str, Any]:
        return {
            "format": "ldeprompt-mindspore-v1",
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "length": self.prompt_length,
            "num_classes": self.num_classes,
            "current_task": self.current_task,
            "class_to_task": self.class_to_task,
            "task_prompts": len(self.task_prompts),
            "selected_layers": sorted(self.prompt_pool.selected_layers),
            "pools": self.prompt_pool.pool_snapshot(),
        }

    def save_bundle(self, prefix: str | Path) -> Dict[str, str]:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = prefix.with_suffix(".ckpt")
        metadata_path = prefix.with_suffix(".json")
        self.canonicalize_parameter_names()
        parameters = {
            parameter.name: parameter
            for full_name, parameter in self.parameters_and_names()
            if ".train_prompts." not in full_name
        }
        ms.save_checkpoint(parameters, str(checkpoint_path))
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(self.state_metadata(), handle, ensure_ascii=False, indent=2)
        return {"checkpoint": str(checkpoint_path), "metadata": str(metadata_path)}

    def restore_bundle(self, prefix: str | Path) -> Dict[str, Any]:
        prefix = Path(prefix)
        with open(prefix.with_suffix(".json"), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        global_counts = {
            int(layer): int(values["global_prompts"])
            for layer, values in metadata.get("pools", {}).items()
        }
        placeholder_counts = {
            int(layer): int(values["placeholders"])
            for layer, values in metadata.get("pools", {}).items()
        }
        self.prompt_pool.restore_counts(global_counts, placeholder_counts)
        while len(self.task_prompts) < int(metadata.get("task_prompts", 1)):
            self._add_task_prompt(len(self.task_prompts))
        for prompt in self.task_prompts:
            prompt.set_trainable(False)
        self.current_task = int(metadata.get("current_task", -1))
        self.class_to_task = [int(value) for value in metadata.get("class_to_task", [])]
        self.canonicalize_parameter_names()
        parameters = ms.load_checkpoint(str(prefix.with_suffix(".ckpt")))
        result = ms.load_param_into_net(self, parameters, strict_load=False)
        self.freeze_backbone()
        return {"loaded": len(parameters), "result": result, "metadata": metadata}

    def canonicalize_parameter_names(self) -> None:
        """Give every parameter a stable, unique name derived from its module path."""
        for full_name, parameter in self.parameters_and_names():
            parameter.name = "param__" + full_name.replace(".", "__")
