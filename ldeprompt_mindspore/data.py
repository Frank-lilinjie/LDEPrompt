"""Dataset and preprocessing pipeline without torchvision dependencies."""

from __future__ import annotations

import logging
import pickle
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _load_cifar100(root: Path, download: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = root / "cifar-100-python"
    archive = root / "cifar-100-python.tar.gz"
    root.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        if not download:
            raise FileNotFoundError(
                f"CIFAR-100 not found at {data_dir}; enable download or set dataset_root"
            )
        logging.info("Downloading CIFAR-100 to %s", archive)
        urllib.request.urlretrieve(CIFAR100_URL, archive)
        with tarfile.open(archive, "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(root, filter="data")
            else:
                tar.extractall(root)

    def _read(split: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(data_dir / split, "rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        images = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.asarray(payload[b"fine_labels"], dtype=np.int64)
        return images, labels

    train_x, train_y = _read("train")
    test_x, test_y = _read("test")
    return train_x, train_y, test_x, test_y


def _folder_split(split_dir: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found under {split_dir}")
    class_names = [path.name for path in class_dirs]
    paths: List[str] = []
    labels: List[int] = []
    for label, class_dir in enumerate(class_dirs):
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        paths.extend(str(path) for path in files)
        labels.extend([label] * len(files))
    return paths, np.asarray(labels, dtype=np.int64), class_names


def _load_image_folder(root: Path) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str]]:
    train_paths, train_labels, train_classes = _folder_split(root / "train")
    test_paths, test_labels, test_classes = _folder_split(root / "test")
    if train_classes != test_classes:
        missing_test = sorted(set(train_classes) - set(test_classes))
        missing_train = sorted(set(test_classes) - set(train_classes))
        raise ValueError(
            "Train/test class folders do not match. "
            f"Missing in test: {missing_test}; missing in train: {missing_train}"
        )
    return train_paths, train_labels, test_paths, test_labels, train_classes


def _make_synthetic_templates(num_classes: int, img_size: int, seed: int) -> np.ndarray:
    """Create deterministic class templates shared by train and test splits."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    templates: List[np.ndarray] = []
    for class_id in range(num_classes):
        angle = 2.0 * np.pi * class_id / max(1, num_classes)
        frequency = 1.0 + class_id % 4
        wave = np.sin(2.0 * np.pi * frequency * (xx * np.cos(angle) + yy * np.sin(angle)) / max(2, img_size))
        base = rng.uniform(-0.8, 0.8, size=3).astype(np.float32)
        tint = rng.uniform(-0.35, 0.35, size=(3, 1, 1)).astype(np.float32)
        template = base[:, None, None] + tint * wave[None, :, :]
        template = 0.5 + 0.35 * np.tanh(template)
        templates.append(template.astype(np.float32))
    return np.stack(templates)


def _make_synthetic_split(
    num_classes: int,
    samples_per_class: int,
    img_size: int,
    noise: float,
    seed: int,
    templates: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if templates is None:
        templates = _make_synthetic_templates(num_classes, img_size, seed)
    if templates.shape != (num_classes, 3, img_size, img_size):
        raise ValueError(
            "Synthetic templates must have shape "
            f"{(num_classes, 3, img_size, img_size)}, got {templates.shape}"
        )
    rng = np.random.RandomState(seed)
    images: List[np.ndarray] = []
    labels: List[int] = []
    for class_id, template in enumerate(templates):
        for _ in range(samples_per_class):
            sample = template + rng.normal(0.0, noise, size=template.shape)
            sample = np.clip(sample, 0.0, 1.0)
            images.append((sample.transpose(1, 2, 0) * 255.0).round().astype(np.uint8))
            labels.append(class_id)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _random_resized_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    area = width * height
    for _ in range(10):
        target_area = area * random.uniform(0.05, 1.0)
        aspect = random.uniform(3.0 / 4.0, 4.0 / 3.0)
        crop_w = int(round(np.sqrt(target_area * aspect)))
        crop_h = int(round(np.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            image = image.crop((left, top, left + crop_w, top + crop_h))
            return image.resize((size, size), Image.Resampling.BILINEAR)
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def _resize_center_crop(image: Image.Image, size: int, resize_short_side: bool) -> Image.Image:
    if resize_short_side:
        width, height = image.size
        scale = 256.0 / min(width, height)
        resized = (max(size, int(round(width * scale))), max(size, int(round(height * scale))))
        if resized != image.size:
            image = image.resize(resized, Image.Resampling.BICUBIC)
    width, height = image.size
    if width < size or height < size:
        canvas = Image.new("RGB", (max(size, width), max(size, height)))
        canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
        image = canvas
        width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _read_image(store: Sequence[Any], use_path: bool, index: int) -> Image.Image:
    value = store[index]
    if use_path:
        with Image.open(str(value)) as image:
            return image.convert("RGB").copy()
    return Image.fromarray(np.asarray(value, dtype=np.uint8)).convert("RGB")


def _preprocess_image(
    image: Image.Image,
    mode: str,
    img_size: int,
    train_augment: bool,
    eval_resize: bool,
    normalize: bool,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    if mode == "train" and train_augment:
        image = _random_resized_crop(image, img_size)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        image = _resize_center_crop(image, img_size, resize_short_side=eval_resize)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if normalize:
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        array = (array - mean_array) / np.maximum(std_array, 1e-6)
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)


class _ImageDataset:
    """Module-level dataset so GeneratorDataset workers can serialize it."""

    def __init__(
        self,
        store: Sequence[Any],
        labels: np.ndarray,
        indices: np.ndarray,
        use_path: bool,
        mode: str,
        img_size: int,
        train_augment: bool,
        eval_resize: bool,
        normalize: bool,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.store = store
        self.labels = labels
        self.indices = indices
        self.use_path = use_path
        self.mode = mode
        self.img_size = int(img_size)
        self.train_augment = bool(train_augment)
        self.eval_resize = bool(eval_resize)
        self.normalize = bool(normalize)
        self.mean = list(mean)
        self.std = list(std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        image = _read_image(self.store, self.use_path, global_index)
        image = _preprocess_image(
            image,
            self.mode,
            self.img_size,
            self.train_augment,
            self.eval_resize,
            self.normalize,
            self.mean,
            self.std,
        )
        return image, np.int32(self.labels[global_index])


class ContinualDataManager:
    """Class-incremental task split and MindSpore data loaders."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = str(config["dataset"]).lower()
        self.root = Path(str(config.get("dataset_root", "data"))).expanduser()
        self.seed = int(config.get("seed", 1993))
        self.img_size = int(config["img_size"])
        self._image_store: Sequence[Any]
        self._use_path = False
        self._load()
        self._finalise_tasks()

    def _load(self) -> None:
        if self.dataset_name == "synthetic":
            num_classes = int(self.config["synthetic_classes"])
            templates = _make_synthetic_templates(
                num_classes,
                self.img_size,
                int(self.config["synthetic_seed"]),
            )
            train_x, train_y = _make_synthetic_split(
                num_classes,
                int(self.config["synthetic_samples_per_class"]),
                self.img_size,
                float(self.config["synthetic_noise"]),
                int(self.config["synthetic_seed"]),
                templates=templates,
            )
            test_x, test_y = _make_synthetic_split(
                num_classes,
                int(self.config["synthetic_eval_samples_per_class"]),
                self.img_size,
                float(self.config["synthetic_noise"]) * 1.25 + 0.01,
                int(self.config["synthetic_seed"]) + 977,
                templates=templates,
            )
            self._train_images = train_x
            self._test_images = test_x
            self._use_path = False
        elif self.dataset_name in {"cifar100", "cifar224"}:
            self._train_images, self._train_labels_raw, self._test_images, self._test_labels_raw = _load_cifar100(
                self.root, bool(self.config.get("download", True))
            )
            self._use_path = False
        elif self.dataset_name in {"cub", "vtab"}:
            train_paths, train_labels, test_paths, test_labels, _ = _load_image_folder(self.root)
            self._train_images = train_paths
            self._test_images = test_paths
            self._train_labels_raw = train_labels
            self._test_labels_raw = test_labels
            self._use_path = True
        else:
            raise NotImplementedError(
                f"Unsupported dataset {self.dataset_name!r}; use synthetic, cifar224, cub, or vtab"
            )

        if not hasattr(self, "_train_labels_raw"):
            self._train_labels_raw = train_y
            self._test_labels_raw = test_y

        original_classes = np.unique(self._train_labels_raw).astype(np.int64)
        class_order = original_classes.copy()
        if bool(self.config.get("shuffle_classes", True)):
            rng = np.random.RandomState(self.seed)
            class_order = class_order[rng.permutation(len(class_order))]
        self.class_order = class_order
        self.original_to_new = {int(old): new for new, old in enumerate(class_order)}
        self.train_labels = np.asarray(
            [self.original_to_new[int(label)] for label in self._train_labels_raw], dtype=np.int64
        )
        self.test_labels = np.asarray(
            [self.original_to_new[int(label)] for label in self._test_labels_raw], dtype=np.int64
        )
        max_train = int(self.config.get("max_samples_per_class", 0) or 0)
        max_eval = int(self.config.get("eval_samples_per_class", 0) or 0)
        self._train_indices = self._balanced_indices(self.train_labels, max_train, self.seed + 31)
        self._test_indices = self._balanced_indices(self.test_labels, max_eval, self.seed + 73)

    @staticmethod
    def _balanced_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
        if maximum <= 0:
            return np.arange(len(labels), dtype=np.int64)
        rng = np.random.RandomState(seed)
        selected: List[int] = []
        for class_id in np.unique(labels):
            candidates = np.flatnonzero(labels == class_id)
            if len(candidates) > maximum:
                candidates = rng.choice(candidates, size=maximum, replace=False)
            selected.extend(int(index) for index in np.sort(candidates))
        return np.asarray(selected, dtype=np.int64)

    def _finalise_tasks(self) -> None:
        num_classes = len(np.unique(self.train_labels))
        init_cls = int(self.config["init_cls"])
        increment = int(self.config["increment"])
        if init_cls > num_classes:
            init_cls = num_classes
        increments = [init_cls]
        while sum(increments) + increment < num_classes:
            increments.append(increment)
        remainder = num_classes - sum(increments)
        if remainder > 0:
            increments.append(remainder)
        max_tasks = int(self.config.get("max_tasks", 0) or 0)
        if max_tasks > 0:
            increments = increments[:max_tasks]
        self.increments = increments
        self.task_classes: List[List[int]] = []
        start = 0
        for size in increments:
            self.task_classes.append(list(range(start, start + size)))
            start += size
        self.seen_classes = start
        self.class_to_task = np.full(start, -1, dtype=np.int32)
        for task_id, classes in enumerate(self.task_classes):
            self.class_to_task[classes] = task_id
        self.num_tasks = len(increments)
        self.nb_classes = start

    @property
    def train_indices(self) -> np.ndarray:
        return self._train_indices

    @property
    def test_indices(self) -> np.ndarray:
        return self._test_indices

    def indices_for(self, source: str, classes: Sequence[int] | None = None) -> np.ndarray:
        if source == "train":
            base = self._train_indices
            labels = self.train_labels
        elif source == "test":
            base = self._test_indices
            labels = self.test_labels
        else:
            raise ValueError(f"Unknown source: {source}")
        if classes is None:
            return base
        class_set = np.asarray(list(classes), dtype=np.int64)
        mask = np.isin(labels[base], class_set)
        return base[mask]

    def class_count(self, source: str, class_id: int) -> int:
        indices = self.indices_for(source, [class_id])
        return int(len(indices))

    def make_dataset(
        self,
        indices: np.ndarray,
        source: str,
        mode: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ):
        import mindspore as ms

        store = self._train_images if source == "train" else self._test_images
        labels = self.train_labels if source == "train" else self.test_labels
        use_path = self._use_path
        dataset_source = _ImageDataset(
            store=store,
            labels=labels,
            indices=indices,
            use_path=use_path,
            mode=mode,
            img_size=self.img_size,
            train_augment=bool(self.config.get("train_augment", True)),
            eval_resize=bool(self.config.get("eval_resize", True)),
            normalize=bool(self.config.get("normalize", False)),
            mean=self.config.get("mean", [0.0, 0.0, 0.0]),
            std=self.config.get("std", [1.0, 1.0, 1.0]),
        )
        workers = max(1, min(10, int(num_workers)))
        dataset = ms.dataset.GeneratorDataset(
            dataset_source,
            column_names=["image", "label"],
            num_samples=len(indices),
            num_parallel_workers=workers,
            shuffle=shuffle,
            python_multiprocessing=workers > 1,
        )
        return dataset.batch(batch_size, drop_remainder=False)

    def train_loader(self, classes: Sequence[int], batch_size: int, shuffle: bool = True, num_workers: int = 0):
        return self.make_dataset(
            self.indices_for("train", classes),
            "train",
            "train",
            batch_size,
            shuffle,
            num_workers,
        )
    def eval_loader(self, classes: Sequence[int], batch_size: int, num_workers: int = 0):
        return self.make_dataset(
            self.indices_for("test", classes),
            "test",
            "test",
            batch_size,
            False,
            num_workers,
        )
    def feature_loader(self, source: str, classes: Sequence[int], batch_size: int, num_workers: int = 0):
        return self.make_dataset(
            self.indices_for(source, classes),
            source,
            "test",
            batch_size,
            False,
            num_workers,
        )
