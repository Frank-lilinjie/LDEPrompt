"""Continual-learning trainer for the MindSpore LDEPrompt port."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import mindspore as ms
from mindspore import nn, ops, value_and_grad

from .data import ContinualDataManager
from .model import LDEPromptViT, select_important_layers
from .prompt import select_reusable_prompt_ids


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_logit_mask(num_classes: int, known_classes: int, total_classes: int) -> np.ndarray:
    """Mask old classes and classes from future tasks during current-task training."""
    mask = np.zeros((int(num_classes),), dtype=np.float32)
    known = max(0, min(int(known_classes), int(num_classes)))
    total = max(known, min(int(total_classes), int(num_classes)))
    mask[:known] = -1.0e4
    mask[total:] = -1.0e4
    return mask


class LDEPromptLearner:
    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: ContinualDataManager,
        output_dir: str | Path,
        runtime_info: Dict[str, Any] | None = None,
    ):
        self.config = dict(config)
        self.data_manager = data_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_info = dict(runtime_info or {})
        self.model = LDEPromptViT(self.config, data_manager.nb_classes)
        self.model.class_to_task = [int(value) for value in data_manager.class_to_task]
        self.known_classes = 0
        self.total_classes = 0
        self.classifier_stats: Dict[int, Dict[str, np.ndarray]] = {}
        self.acc_matrix = np.zeros((data_manager.num_tasks, data_manager.num_tasks), dtype=np.float64)
        self.top1_curve: List[float] = []
        self.per_task_curves: List[List[float]] = []
        self.selected_layers_by_task: Dict[int, List[int]] = {}
        self.information_gains_by_task: Dict[int, List[float]] = {}
        self.reused_prompts_by_task: Dict[int, Dict[int, List[int]]] = {}
        self.new_prompt_counts_by_task: Dict[int, Dict[int, int]] = {}
        checkpoint = str(self.config.get("backbone_checkpoint", "") or "").strip()
        if checkpoint:
            logging.info("Loading backbone checkpoint: %s", checkpoint)
            self.model.load_backbone_checkpoint(checkpoint)

    @property
    def batch_size(self) -> int:
        return int(self.config["batch_size"])

    @property
    def num_workers(self) -> int:
        return int(self.config.get("num_workers", 0) or 0)

    def run(self) -> Dict[str, Any]:
        started = time.time()
        for task_id, classes in enumerate(self.data_manager.task_classes):
            logging.info("=" * 72)
            logging.info("Task %d classes %s", task_id, classes)
            self.incremental_train(task_id, classes)
            accuracies, overall = self.evaluate_seen_tasks(task_id)
            self.acc_matrix[task_id, : task_id + 1] = accuracies
            self.top1_curve.append(overall)
            self.per_task_curves.append(accuracies)
            logging.info("Task %d per-task accuracies: %s", task_id, accuracies)
            logging.info("Task %d seen-class top1: %.2f", task_id, overall)
        metrics = self.build_metrics(time.time() - started)
        self.save_outputs(metrics)
        return metrics

    def incremental_train(self, task_id: int, classes: Sequence[int]) -> None:
        self.total_classes = self.known_classes + len(classes)
        self.model.prepare_task_prompts(task_id)
        gain_loader = self.data_manager.train_loader(
            classes, self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        gains = self.model.information_gain_scores(
            gain_loader, max_batches=int(self.config.get("importance_max_batches", 0) or 0)
        )
        selected_layers = select_important_layers(
            gains,
            normalization=str(self.config.get("importance_normalization", "zscore")),
            depth=self.model.depth,
        )
        logging.info("Selected prompt layers for task %d: %s", task_id, selected_layers)

        reused, add_counts = self.select_prompts_for_training(task_id, classes, selected_layers)
        self.model.prompt_pool.prepare_task(selected_layers, reused, add_counts)
        self.selected_layers_by_task[task_id] = selected_layers
        self.information_gains_by_task[task_id] = gains
        self.reused_prompts_by_task[task_id] = reused
        self.new_prompt_counts_by_task[task_id] = add_counts

        optimizer, steps_per_epoch = self._make_optimizer(task_id)
        self._train_classifier(task_id, classes, optimizer, steps_per_epoch)
        self.model.merge_task_prompts()

        current_loader = self.data_manager.feature_loader(
            "train", classes, self.batch_size, self.num_workers
        )
        self.model.update_task_prototypes(current_loader, task_id)
        self._compute_classifier_statistics(task_id, classes)
        if task_id > 0 and int(self.config.get("crct_epochs", 0)) > 0:
            self.align_classifier(task_id)
        self.known_classes = self.total_classes

    def _make_optimizer(self, task_id: int):
        parameters = [
            parameter
            for parameter in self.model.trainable_task_parameters(task_id)
            if parameter.requires_grad
        ]
        parameters = list({id(parameter): parameter for parameter in parameters}.values())
        self.model.canonicalize_parameter_names()
        if not parameters:
            raise RuntimeError("No trainable parameters for the current task")
        logging.info("Trainable tensors for task %d: %d", task_id, len(parameters))
        steps_per_epoch = max(
            1,
            math.ceil(
                len(self.data_manager.indices_for("train", self.data_manager.task_classes[task_id]))
                / float(self.batch_size)
            ),
        )
        total_steps = max(1, steps_per_epoch * int(self.config["tuned_epoch"]))
        init_lr = float(self.config["init_lr"])
        scheduler = str(self.config.get("scheduler", "cosine")).lower()
        learning_rate: Any = init_lr
        if scheduler == "cosine":
            learning_rate = nn.CosineDecayLR(
                min_lr=float(self.config["min_lr"]),
                max_lr=init_lr,
                decay_steps=total_steps,
            )
        elif scheduler == "steplr":
            learning_rate = init_lr
        elif scheduler != "constant":
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        name = str(self.config.get("optimizer", "adamw")).lower()
        weight_decay = float(self.config.get("weight_decay", 0.0) or 0.0)
        if name == "sgd":
            optimizer = nn.SGD(
                parameters,
                learning_rate=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif name == "adam":
            optimizer = nn.Adam(parameters, learning_rate=learning_rate, weight_decay=weight_decay)
        elif name == "adamw":
            optimizer = nn.AdamWeightDecay(
                parameters, learning_rate=learning_rate, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {name}")
        return optimizer, steps_per_epoch

    def _train_classifier(self, task_id: int, classes: Sequence[int], optimizer, steps_per_epoch: int) -> None:
        mask = build_logit_mask(self.model.num_classes, self.known_classes, self.total_classes)
        logit_mask = ms.Tensor(mask.reshape(1, -1))
        diversity_coeff = float(self.config.get("diversity_loss_coeff", 0.0))
        placeholder_coeff = float(self.config.get("placeholder_loss_coeff", 0.0))
        selected = self.model.prompt_pool.selected_layers
        add_counts = self.new_prompt_counts_by_task[task_id]
        trainable = [
            parameter
            for parameter in self.model.trainable_task_parameters(task_id)
            if parameter.requires_grad
        ]
        trainable = list({id(parameter): parameter for parameter in trainable}.values())

        def loss_function(images, labels):
            output = self.model.forward_train(images, task_id)
            logits = output["logits"] + logit_mask
            classification = ops.cross_entropy(logits, labels.astype(ms.int32))
            diversity = self.model.prompt_diversity_loss(selected, add_counts)
            placeholder = self.model.placeholder_orthogonality_loss(selected)
            loss = classification - diversity_coeff * diversity + placeholder_coeff * placeholder
            return loss, (classification, diversity, placeholder, logits)

        grad_fn = value_and_grad(
            loss_function, grad_position=None, weights=trainable, has_aux=True
        )
        scheduler = str(self.config.get("scheduler", "cosine")).lower()
        milestones = [int(value) for value in self.config.get("init_milestones", [])]
        lr_decay = float(self.config.get("init_lr_decay", 0.5))
        for epoch in range(int(self.config["tuned_epoch"])):
            loader = self.data_manager.train_loader(
                classes, self.batch_size, shuffle=True, num_workers=self.num_workers
            )
            total_loss = 0.0
            total_cls = 0.0
            total_div = 0.0
            total_placeholder = 0.0
            correct = 0
            seen = 0
            batches = 0
            for images, labels in loader.create_tuple_iterator():
                (loss, auxiliary), gradients = grad_fn(images, labels)
                optimizer(gradients)
                classification, diversity, placeholder, logits = auxiliary
                batch = int(images.shape[0])
                total_loss += float(loss.asnumpy()) * batch
                total_cls += float(classification.asnumpy()) * batch
                total_div += float(diversity.asnumpy()) * batch
                total_placeholder += float(placeholder.asnumpy()) * batch
                predictions = ops.argmax(logits, 1).asnumpy()
                correct += int((predictions == labels.asnumpy()).sum())
                seen += batch
                batches += 1
            if scheduler == "steplr" and epoch + 1 in milestones:
                new_lr = float(optimizer.learning_rate.asnumpy()) * lr_decay
                optimizer.learning_rate.set_data(ms.Tensor(new_lr, ms.float32))
            if seen == 0:
                raise RuntimeError("Training loader yielded no samples")
            logging.info(
                "Task %d epoch %d/%d loss=%.4f ce=%.4f div=%.4f plc=%.4f acc=%.2f",
                task_id,
                epoch + 1,
                int(self.config["tuned_epoch"]),
                total_loss / seen,
                total_cls / seen,
                total_div / seen,
                total_placeholder / seen,
                100.0 * correct / seen,
            )

    def select_prompts_for_training(
        self, task_id: int, classes: Sequence[int], selected_layers: Sequence[int]
    ):
        pool_size = int(self.config["pool_size"])
        reused: Dict[int, List[int]] = {int(layer): [] for layer in selected_layers}
        add_counts: Dict[int, int] = {}
        has_global = {
            int(layer): self.model.prompt_pool.has_global_pool(int(layer))
            for layer in selected_layers
        }
        feature_store: Dict[int, List[np.ndarray]] = {
            int(layer): [] for layer in selected_layers if has_global[int(layer)]
        }
        if feature_store:
            loader = self.data_manager.train_loader(
                classes, self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            for batch in loader.create_tuple_iterator():
                outputs = self.model.features_at_layers(batch[0], feature_store.keys())
                for layer, value in outputs.items():
                    feature_store[layer].append(value.asnumpy())
        for layer in selected_layers:
            layer = int(layer)
            if has_global[layer]:
                features = np.concatenate(feature_store[layer], axis=0).astype(np.float64)
                keys = np.concatenate(
                    [prompt.prompt_key.asnumpy() for prompt in self.model.prompt_pool.layer(layer).global_prompts],
                    axis=0,
                ).astype(np.float64)
                feature_norm = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
                key_norm = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)
                similarity = feature_norm @ key_norm.T
                mean_similarity = similarity.mean(axis=0)
                max_old = max(0, pool_size - 1)
                reused[layer] = select_reusable_prompt_ids(mean_similarity, max_old)
            add_counts[layer] = pool_size - len(reused[layer])
            logging.info(
                "Task %d layer %d reuses %s, adds %d prompts",
                task_id,
                layer,
                reused[layer],
                add_counts[layer],
            )
        return reused, add_counts

    def _compute_classifier_statistics(self, task_id: int, classes: Sequence[int]) -> None:
        loader = self.data_manager.feature_loader(
            "train", classes, self.batch_size, self.num_workers
        )
        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for images, batch_labels in loader.create_tuple_iterator():
            task_ids = [int(task_id)] * int(images.shape[0])
            tokens = self.model.forward_features_eval(images, task_ids)
            features.append(tokens[:, 0].asnumpy().astype(np.float64))
            labels.append(batch_labels.asnumpy().astype(np.int64))
        all_features = np.concatenate(features, axis=0)
        all_labels = np.concatenate(labels, axis=0)
        method = str(self.config.get("ca_storage_efficient_method", "covariance"))
        for class_id in classes:
            data = all_features[all_labels == int(class_id)]
            if len(data) == 0:
                raise RuntimeError(f"No features extracted for class {class_id}")
            mean = data.mean(axis=0)
            stats: Dict[str, np.ndarray] = {"mean": mean.astype(np.float32)}
            if method == "covariance":
                if len(data) > 1:
                    covariance = np.cov(data, rowvar=False)
                else:
                    covariance = np.eye(data.shape[1], dtype=np.float64)
                covariance = np.atleast_2d(covariance)
                covariance = covariance + np.eye(data.shape[1], dtype=np.float64) * 1e-4
                stats["covariance"] = covariance.astype(np.float32)
            elif method == "variance":
                variance = data.var(axis=0) if len(data) > 1 else np.ones(data.shape[1])
                stats["variance"] = (variance + 1e-4).astype(np.float32)
            elif method == "multi-centroid":
                centroids, variances = self._multi_centroid(data)
                stats["centroids"] = centroids.astype(np.float32)
                stats["variances"] = variances.astype(np.float32)
            else:
                raise ValueError(f"Unknown statistics method: {method}")
            self.classifier_stats[int(class_id)] = stats
        logging.info("Computed classifier statistics for %d classes", len(classes))

    def _multi_centroid(self, data: np.ndarray):
        n_clusters = min(int(self.config.get("n_centroids", 10)), len(data))
        if n_clusters <= 1:
            return data.mean(axis=0, keepdims=True), (data.var(axis=0) + 1e-4)[None, :]
        rng = np.random.RandomState(int(self.config.get("seed", 1993)) + len(data))
        centers = data[rng.choice(len(data), size=n_clusters, replace=False)]
        assignments = np.zeros(len(data), dtype=np.int64)
        for _ in range(10):
            distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assignments = distances.argmin(axis=1)
            for index in range(n_clusters):
                members = data[assignments == index]
                if len(members):
                    centers[index] = members.mean(axis=0)
        centroids: List[np.ndarray] = []
        variances: List[np.ndarray] = []
        for index in range(n_clusters):
            members = data[assignments == index]
            if len(members) == 0:
                members = centers[index : index + 1]
            centroids.append(members.mean(axis=0))
            variances.append(members.var(axis=0) + 1e-4)
        return np.stack(centroids), np.stack(variances)

    def _sample_classifier_features(self, classes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        method = str(self.config.get("ca_storage_efficient_method", "covariance"))
        samples_per_class = int(self.config.get("ca_samples_per_class", 120))
        rng = np.random.RandomState(int(self.config.get("seed", 1993)) + self.total_classes + 17)
        feature_batches: List[np.ndarray] = []
        label_batches: List[np.ndarray] = []
        for class_id in classes:
            stats = self.classifier_stats[int(class_id)]
            if method == "covariance":
                values = rng.multivariate_normal(
                    stats["mean"].astype(np.float64),
                    stats["covariance"].astype(np.float64),
                    size=samples_per_class,
                    check_valid="ignore",
                )
            elif method == "variance":
                values = rng.normal(
                    stats["mean"],
                    np.sqrt(stats["variance"]),
                    size=(samples_per_class, len(stats["mean"])),
                )
            elif method == "multi-centroid":
                centroids = stats["centroids"]
                variances = stats["variances"]
                per_cluster = max(1, samples_per_class // len(centroids))
                values = np.concatenate(
                    [
                        rng.normal(mean, np.sqrt(var), size=(per_cluster, len(mean)))
                        for mean, var in zip(centroids, variances)
                    ],
                    axis=0,
                )
            else:
                raise ValueError(f"Unknown statistics method: {method}")
            feature_batches.append(values.astype(np.float32))
            label_batches.append(np.full((len(values),), int(class_id), dtype=np.int32))
        return np.concatenate(feature_batches), np.concatenate(label_batches)

    def align_classifier(self, task_id: int) -> None:
        features, labels = self._sample_classifier_features(range(self.total_classes))
        head_parameters = list(self.model.head.trainable_params())
        optimizer = nn.Adam(head_parameters, learning_rate=float(self.config["ca_lr"]))
        feature_tensor = ms.Tensor(features, ms.float32)
        label_tensor = ms.Tensor(labels, ms.int32)
        batch_size = int(self.config.get("ca_batch_size", 256))

        def loss_function(batch_features, batch_labels):
            logits = self.model.head(batch_features)[:, : self.total_classes]
            return ops.cross_entropy(logits, batch_labels), logits

        grad_fn = value_and_grad(
            loss_function, grad_position=None, weights=head_parameters, has_aux=True
        )
        for epoch in range(int(self.config.get("crct_epochs", 0))):
            indices = np.arange(len(features))
            np.random.RandomState(int(self.config.get("seed", 1993)) + epoch).shuffle(indices)
            correct = 0
            total = 0
            total_loss = 0.0
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                index_tensor = ms.Tensor(batch_indices, ms.int32)
                batch_features = ops.gather(feature_tensor, index_tensor, 0)
                batch_labels = ops.gather(label_tensor, index_tensor, 0)
                (loss, logits), gradients = grad_fn(batch_features, batch_labels)
                optimizer(gradients)
                predictions = ops.argmax(logits, 1).asnumpy()
                correct += int((predictions == batch_labels.asnumpy()).sum())
                total += len(batch_indices)
                total_loss += float(loss.asnumpy()) * len(batch_indices)
            logging.info(
                "Classifier alignment task %d epoch %d/%d loss=%.4f acc=%.2f",
                task_id,
                epoch + 1,
                int(self.config["crct_epochs"]),
                total_loss / max(1, total),
                100.0 * correct / max(1, total),
            )

    def evaluate_seen_tasks(self, task_id: int):
        accuracies: List[float] = []
        all_correct = 0
        all_total = 0
        for group_task in range(task_id + 1):
            classes = self.data_manager.task_classes[group_task]
            loader = self.data_manager.eval_loader(classes, self.batch_size, self.num_workers)
            correct = 0
            total = 0
            for images, labels in loader.create_tuple_iterator():
                output = self.model.forward_eval(images, num_classes=self.total_classes)
                logits = output["logits"][:, : self.total_classes]
                predictions = ops.argmax(logits, 1).asnumpy()
                correct += int((predictions == labels.asnumpy()).sum())
                total += int(images.shape[0])
            accuracy = 100.0 * correct / max(1, total)
            accuracies.append(round(float(accuracy), 2))
            all_correct += correct
            all_total += total
        return accuracies, round(100.0 * all_correct / max(1, all_total), 2)

    def build_metrics(self, elapsed_seconds: float) -> Dict[str, Any]:
        task_id = self.data_manager.num_tasks - 1
        final = self.acc_matrix[task_id, : task_id + 1]
        average_accuracy = float(final.mean()) if len(final) else 0.0
        last_accuracy = float(final[-1]) if len(final) else 0.0
        forgetting = 0.0
        if task_id > 0:
            differences = []
            for group in range(task_id):
                best = float(self.acc_matrix[: task_id, group].max())
                differences.append(best - float(self.acc_matrix[task_id, group]))
            forgetting = float(np.mean(differences)) if differences else 0.0
        return {
            "dataset": self.config.get("dataset"),
            "device": self.runtime_info,
            "num_tasks": self.data_manager.num_tasks,
            "class_increments": self.data_manager.increments,
            "task_classes": self.data_manager.task_classes,
            "top1_curve": self.top1_curve,
            "per_task_curves": self.per_task_curves,
            "accuracy_matrix": self.acc_matrix[: self.data_manager.num_tasks, : self.data_manager.num_tasks],
            "average_accuracy": round(average_accuracy, 2),
            "last_accuracy": round(last_accuracy, 2),
            "forgetting": round(forgetting, 2),
            "selected_layers": self.selected_layers_by_task,
            "information_gains": self.information_gains_by_task,
            "reused_prompt_ids": self.reused_prompts_by_task,
            "new_prompt_counts": self.new_prompt_counts_by_task,
            "prompt_pools": self.model.prompt_pool.pool_snapshot(),
            "elapsed_seconds": round(float(elapsed_seconds), 2),
        }

    def save_outputs(self, metrics: Dict[str, Any]) -> None:
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(jsonable(metrics), handle, ensure_ascii=False, indent=2)
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(jsonable(self.config), handle, ensure_ascii=False, indent=2)
        if bool(self.config.get("save_checkpoint", True)):
            bundle = self.model.save_bundle(self.output_dir / "model")
            logging.info("Saved model bundle: %s", bundle)
