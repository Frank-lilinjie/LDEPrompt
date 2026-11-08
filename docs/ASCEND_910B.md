# LDEPrompt 昇腾 910B（MindSpore/CANN）适配说明

本文档对应当前分支 `feature/昇腾910B-NPU适配`。该分支保留了原始 PyTorch 实现，同时新增了一套 MindSpore/CANN 单卡实现，用于在华为昇腾 910B 上运行 LDEPrompt。

## 1. 适配范围

本分支提供：

- ViT-B/16 主干、Patch Embedding、Attention、MLP 和 LayerNorm 的 MindSpore 实现。
- g-prompt、task prompt、层次 Prompt 池、占位符和 Prompt 复用/扩展逻辑。
- 基于中间层信息熵变化的自适应选层。
- 任务的增量训练、任务原型路由、分类器统计量估计和分类器对齐。
- MindSpore checkpoint/元数据保存与恢复。
- 论文 JAX `.npz` ViT 权重到 MindSpore `.ckpt` 的转换脚本。
- 22 项单元测试，以及小模型三任务 smoke。

运行环境为与主机驱动匹配的 CANN 和 MindSpore Ascend wheel。数据和 ViT 权重按实验配置提供；训练入口面向 Ascend 910B 单卡。

## 2. 实现结构

| 原 PyTorch 组件 | MindSpore 实现 | 说明 |
| --- | --- | --- |
| `backbone/vit_ldeprompt.py` | `ldeprompt_mindspore/model.py` | ViT-B/16、Attention、Prompt 注入 |
| `backbone/prompt.py` | `ldeprompt_mindspore/prompt.py` | 层次 Prompt 池、路由 key、复用和扩展 |
| `models/ldeprompt.py` | `ldeprompt_mindspore/trainer.py` | 增量任务训练、选层、分类器对齐 |
| `utils/data.py` | `ldeprompt_mindspore/data.py` | CIFAR/CUB/VTAB 读取与预处理 |
| PyTorch 优化器/loss | `mindspore.nn`、`value_and_grad` | 不依赖 Torch/TorchNPU |
| Torch checkpoint | MindSpore `.ckpt` + JSON 元数据 | 由 `save_bundle`/`restore_bundle` 管理 |

入口：

- 论文原 PyTorch 入口：`main.py`，保持原样。
- 昇腾适配入口：`main_mindspore.py`。
- Python 包方式：`python -m ldeprompt_mindspore`。

默认使用 `PYNATIVE_MODE`。Prompt 池在任务之间动态扩张，训练池和路由池包含 Python 侧结构；`--ms-mode graph` 启用 `GRAPH_MODE`。

## 3. 软硬件依赖

建议环境：

- Ascend 910B，单卡。
- 与服务器驱动匹配的 CANN Toolkit。
- 与 CANN 版本匹配的 `mindspore-ascend` wheel。
- Python 3.9-3.11。
- NumPy、Pillow。

MindSpore 使用与主机 CANN 和驱动匹配的 **Ascend 版本 wheel**。具体 wheel 与 CANN 的版本对应关系以华为昇腾官方安装文档和服务器驱动为准。

先检查设备：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "$ASCEND_HOME_PATH"
```

检查 Python 环境：

```bash
python -c "import mindspore as ms; print(ms.__version__)"
pip show mindspore-ascend
```

检查 MindSpore 能否初始化 Ascend：

```bash
python - <<'PY'
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
print("MindSpore:", ms.__version__)
print("Target:", ms.get_context("device_target"))
PY
```

安装本仓库的 Python 依赖：

```bash
python -m pip install -r requirements-mindspore.txt
```

`requirements-mindspore.txt` 提供 MindSpore 之外的 Python 依赖；MindSpore 本体按 CANN 版本单独安装 Ascend wheel。

## 4. 数据集

### CIFAR-100

`exps/ascend/cifar/ldeprompt.json` 默认：

```text
dataset_root = data
download = true
```

首次运行会下载 CIFAR-100 到 `data/cifar-100-python.tar.gz` 并解压。离线机器请预先放好该目录并关闭下载。

### CUB-200 / VTAB

配置默认假设如下目录结构：

```text
data/cub/
├── train/
│   ├── class_000/
│   │   ├── image_0001.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class_000/
    └── ...
```

VTAB 同样使用 `train/<class>/*` 与 `test/<class>/*`。`train` 和 `test` 的类别文件夹必须一致。可通过以下方式覆盖路径：

```bash
CONFIG=exps/ascend/cub/ldeprompt.json \
DATASET_ROOT=/path/to/cub \
bash scripts/run_ascend_910b.sh --dataset-root /path/to/cub
```

脚本也支持直接传 `--dataset-root`，该参数优先级最高。

## 5. 预训练 ViT 权重转换

论文实现使用 Google JAX 格式的 ViT-B/16 权重，原 PyTorch 代码通过 timm 加载 `.npz`。MindSpore 版本需要先转换为 `.ckpt`。

仓库中的原实现对应 ImageNet-1K 权重地址：

```text
https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
```

下载后执行：

```bash
python scripts/convert_jax_vit.py \
  --source /path/to/B_16-imagenet.npz \
  --output /path/to/vit_b16_in1k.ckpt
```

如果使用 ImageNet-21K 权重，也使用相同脚本。转换器默认 `--prefix auto`，会识别常见的 `opt/target/` 前缀；如文件使用其他前缀，可显式指定：

```bash
python scripts/convert_jax_vit.py \
  --source /path/to/weights.npz \
  --output /path/to/vit_b16_in21k.ckpt \
  --prefix opt/target/
```

转换会处理：

- JAX `NHWC`/卷积布局到 MindSpore `NCHW`/卷积权重的转置。
- 分开的 query/key/value 权重到合并 QKV 权重的拼接。
- JAX LayerNorm `scale/bias` 到 MindSpore `gamma/beta` 名称映射。
- 位置编码、CLS token、patch projection、MLP 和输出层归一化。

ViT-B/16（12 个 block、QKV bias 开启）转换后应有 150 个主干张量。加载时会打印实际加载和跳过的数量；`skipped` 应接近 0，若不为 0 请检查 backbone 配置是否与权重一致。

运行示例：

```bash
CONFIG=exps/ascend/cifar/ldeprompt.json \
BACKBONE_CHECKPOINT=/path/to/vit_b16_in1k.ckpt \
OUTPUT_DIR=runs/ascend-cifar \
bash scripts/run_ascend_910b.sh --seed 1993
```

`BACKBONE_CHECKPOINT` 指向与配置匹配的 ViT 预训练权重；未设置时使用模型默认初始化。

## 6. 910B 启动

脚本 `scripts/run_ascend_910b.sh` 会：

- 尝试 source `$ASCEND_HOME_PATH/set_env.sh` 或常见 CANN 路径下的环境脚本。
- 检查 `ASCEND_HOME_PATH` 和 `npu-smi`。
- 打印设备信息。
- 以 `npu:<DEVICE_ID>`、`--device-target ascend` 启动 MindSpore。

先执行小型 smoke：

```bash
CONFIG=exps/smoke/ldeprompt_synthetic.json \
DEVICE_ID=0 \
OUTPUT_DIR=runs/ascend-smoke \
bash scripts/run_ascend_910b.sh
```

该配置使用 32×32 输入、4 层小 ViT、3 个任务，覆盖设备初始化、训练、评估和 checkpoint 路径；论文实验使用 `exps/ascend/` 下的完整配置。

只跑 CIFAR 的第一个任务：

```bash
CONFIG=exps/ascend/cifar/ldeprompt.json \
BACKBONE_CHECKPOINT=/path/to/vit_b16_in1k.ckpt \
OUTPUT_DIR=runs/ascend-cifar-task0 \
bash scripts/run_ascend_910b.sh --max-tasks 1 --seed 1993
```

运行完整配置：

```bash
CONFIG=exps/ascend/cifar/ldeprompt.json \
BACKBONE_CHECKPOINT=/path/to/vit_b16_in1k.ckpt \
OUTPUT_DIR=runs/ascend-cifar-full \
bash scripts/run_ascend_910b.sh
```

常用环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `PYTHON_BIN` | `python3` | Python 解释器 |
| `DEVICE_ID` | `0` | 使用的 910B 卡号 |
| `CONFIG` | CIFAR 配置 | 实验 JSON |
| `OUTPUT_DIR` | `runs/<EXP>` | 输出目录 |
| `MS_MODE` | `pynative` | MindSpore 执行模式 |
| `BACKBONE_CHECKPOINT` | 空 | 转换后的 ViT `.ckpt` |

也可以绕过脚本直接运行：

```bash
python main_mindspore.py \
  --config exps/ascend/cifar/ldeprompt.json \
  --device npu:0 \
  --device-target ascend \
  --ms-mode pynative \
  --backbone-checkpoint /path/to/vit_b16_in1k.ckpt \
  --output-dir runs/ascend-cifar
```

## 7. CPU smoke（可选）

在 CPU 环境运行同一套 MindSpore 训练路径：

```bash
PYTHON_BIN=python3 bash scripts/run_mindspore_smoke.sh
```

或直接执行：

```bash
python main_mindspore.py \
  --config exps/smoke/ldeprompt_synthetic.json \
  --device cpu \
  --device-target cpu \
  --ms-mode pynative \
  --output-dir runs/smoke
```

单元测试：

```bash
python -m unittest discover -s tests -v
```

## 8. 输出文件

每个实验目录包含：

```text
runs/<experiment>/
├── train.log
├── run.json
├── metrics.json
├── config.json
├── model.ckpt
└── model.json
```

`metrics.json` 记录任务类别、选层结果、信息增益、复用 Prompt、准确率矩阵、平均准确率、遗忘率和运行设备信息。`model.json` 保存 Prompt 池大小、选层、类别到任务映射等恢复所需元数据。

## 9. 910B 资源建议

- 单任务运行示例：`--max-tasks 1 --seed 1993`。
- ViT-B/16 默认 `batch_size=24`；若出现显存不足，可先降到 12 或 8，再观察吞吐。
- `num_workers` 默认 4；如果 CPU 预处理成为瓶颈，可尝试 8，若内存压力大则降为 2。
- Ascend 910B 运行时使用 PyNative，并关闭其他占卡进程。
- 数据增强和预处理在 CPU 侧执行；吞吐由主机 CPU 和存储读取速度决定。
- 训练入口按 Ascend 910B 单卡组织。

## 10. 算法语义说明

MindSpore 版本按发布代码记录算法语义：

- 前向计算信息熵，使用相邻 block 前后熵差的绝对值作为信息增益。
- z-score 的均值与标准差只由中间层计算，首层和末层不参与统计。
- 选择归一化分数大于 0 的中间层；若没有层满足条件，回退到增益最大的中间层。
- Prompt 复用规则为：先保留正相似度，再取高于这些正相似度均值的旧 Prompt，最多 `pool_size - 1` 个，以确保当前任务至少新增一个 Prompt。
- 新训练完成的 Prompt 追加到全局池，旧全局 Prompt 冻结；这是原实现的“只追加新增项”语义。

实验对照同时使用论文公式、公开代码和对应配置文件。

## 11. 执行模式与排障

### 执行模式

优先使用：

```bash
--ms-mode pynative
```

`GRAPH_MODE` 通过 `--ms-mode graph` 启用；图编译行为由 MindSpore 版本、CANN 算子覆盖和动态 shape 配置决定。

### 预训练权重检查

检查日志中的 `Loaded ... backbone tensors; skipped ...`。若 `skipped > 0`，常见原因是：

- 配置的 `embed_dim/depth/num_heads/patch_size` 与权重不是同一个 ViT。
- `.npz` 使用了非标准前缀。
- 转换脚本输出的 key 与 MindSpore 参数名不一致。

### 显存不足

减小 `batch_size`、`num_workers`，或使用 `--max-tasks 1` 运行单任务。ViT 位置编码与 `img_size/patch_size` 匹配。

### 找不到 CANN 环境

手工执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

然后再运行启动脚本。若 CANN 安装在其他目录，请设置 `ASCEND_HOME_PATH` 并确保 `npu-smi` 在 `PATH` 中。

### 数据集路径错误

CUB/VTAB 必须同时存在 `train/<class>` 和 `test/<class>`，且两边类别目录名完全一致。CIFAR 首次下载需要网络写权限。
