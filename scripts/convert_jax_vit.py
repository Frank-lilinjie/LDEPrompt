#!/usr/bin/env python3
"""Convert the paper's JAX ViT-B/16 ``.npz`` weights to a MindSpore checkpoint."""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np


def _array(value, transpose: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 4 and array.shape[:3] == (1, 1, 1):
        array = array.reshape(-1)
    if transpose:
        if array.ndim == 4:
            array = array.transpose(3, 2, 0, 1)
        elif array.ndim == 3:
            array = array.transpose(2, 0, 1)
        elif array.ndim == 2:
            array = array.transpose(1, 0)
    return np.ascontiguousarray(array)


def _detect_prefix(arrays, requested: str) -> str:
    if requested != "auto":
        return requested if not requested or requested.endswith("/") else requested + "/"
    if "opt/target/embedding/kernel" in arrays:
        return "opt/target/"
    return ""


def convert_arrays(arrays, prefix: str = "auto", verbose: bool = True) -> OrderedDict:
    prefix = _detect_prefix(arrays, prefix)
    source_names = arrays.files if hasattr(arrays, "files") else list(arrays.keys())

    def get(name: str):
        key = prefix + name
        if key not in arrays:
            raise KeyError(f"Missing JAX weight {key!r}")
        return arrays[key]

    output = OrderedDict()
    patch = _array(get("embedding/kernel"))
    dim = int(patch.shape[0])
    output["patch_embed.proj.weight"] = patch
    output["patch_embed.proj.bias"] = _array(get("embedding/bias"), transpose=False)
    output["cls_token"] = _array(get("cls"), transpose=False)
    output["pos_embed"] = _array(
        get("Transformer/posembed_input/pos_embedding"), transpose=False
    )
    output["norm.gamma"] = _array(get("Transformer/encoder_norm/scale"), transpose=False)
    output["norm.beta"] = _array(get("Transformer/encoder_norm/bias"), transpose=False)

    block_ids = sorted(
        {
            int(match.group(1))
            for key in source_names
            if (match := re.search(r"Transformer/encoderblock_(\d+)/LayerNorm_0/scale$", key))
        }
    )
    if not block_ids:
        raise ValueError("No Transformer encoder blocks found in the source checkpoint")

    for index in block_ids:
        source = f"Transformer/encoderblock_{index}/"
        attention = source + "MultiHeadDotProductAttention_1/"
        target = f"blocks.{index}."

        output[target + "norm1.gamma"] = _array(
            get(source + "LayerNorm_0/scale"), transpose=False
        )
        output[target + "norm1.beta"] = _array(
            get(source + "LayerNorm_0/bias"), transpose=False
        )

        qkv_weights = []
        qkv_biases = []
        for component in ("query", "key", "value"):
            kernel = np.asarray(get(attention + component + "/kernel"), dtype=np.float32)
            qkv_weights.append(kernel.reshape(kernel.shape[0], -1).T)
            qkv_biases.append(
                np.asarray(get(attention + component + "/bias"), dtype=np.float32).reshape(-1)
            )
        output[target + "attn.qkv.weight"] = np.ascontiguousarray(
            np.concatenate(qkv_weights, axis=0)
        )
        output[target + "attn.qkv.bias"] = np.ascontiguousarray(
            np.concatenate(qkv_biases, axis=0)
        )
        output[target + "attn.proj.weight"] = _array(
            get(attention + "out/kernel")
        ).reshape(dim, dim)
        output[target + "attn.proj.bias"] = _array(
            get(attention + "out/bias"), transpose=False
        )

        for number in range(2):
            output[target + f"mlp.fc{number + 1}.weight"] = _array(
                get(source + f"MlpBlock_3/Dense_{number}/kernel")
            )
            output[target + f"mlp.fc{number + 1}.bias"] = _array(
                get(source + f"MlpBlock_3/Dense_{number}/bias"), transpose=False
            )
        output[target + "norm2.gamma"] = _array(
            get(source + "LayerNorm_2/scale"), transpose=False
        )
        output[target + "norm2.beta"] = _array(
            get(source + "LayerNorm_2/bias"), transpose=False
        )

    qkv_shape = output["blocks.0.attn.qkv.weight"].shape
    if qkv_shape != (3 * dim, dim):
        raise ValueError(f"Unexpected QKV shape after conversion: {qkv_shape}")
    if verbose:
        print(f"Converted {len(output)} tensors from {len(block_ids)} encoder blocks")
    return output


def convert_file(source: str | Path, output: str | Path, prefix: str = "auto") -> int:
    import mindspore as ms

    source = Path(source).expanduser()
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source) as arrays:
        converted = convert_arrays(arrays, prefix=prefix)
    parameters = {
        name: ms.Parameter(ms.Tensor(value), name=name)
        for name, value in converted.items()
    }
    ms.save_checkpoint(parameters, str(output))
    print(f"Saved {len(parameters)} backbone tensors to {output}")
    return len(parameters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original JAX ViT .npz checkpoint")
    parser.add_argument("--output", required=True, help="Destination MindSpore .ckpt")
    parser.add_argument(
        "--prefix",
        default="auto",
        help="JAX key prefix, normally auto or opt/target/ (default: auto)",
    )
    args = parser.parse_args()
    convert_file(args.source, args.output, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
