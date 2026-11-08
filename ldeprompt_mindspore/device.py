"""MindSpore context and Ascend device helpers."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Tuple


def _auto_target() -> str:
    backend = os.environ.get("LDEPROMPT_BACKEND", "").strip().lower()
    if backend in {"npu", "ascend", "cann"}:
        return "Ascend"
    if backend == "cpu":
        return "CPU"
    if os.environ.get("ASCEND_HOME_PATH"):
        return "Ascend"
    if os.path.isdir("/usr/local/Ascend/ascend-toolkit"):
        return "Ascend"
    return "CPU"


def parse_device_spec(device: str, requested_target: str = "auto") -> Tuple[str, int]:
    """Return ``(device_target, device_id)`` for an Ascend-friendly device spec."""
    value = str(device).strip()
    target_hint = str(requested_target).strip().lower()
    aliases = {"ascend": "Ascend", "npu": "Ascend", "cann": "Ascend", "cpu": "CPU"}

    if target_hint in aliases:
        forced_target = aliases[target_hint]
    elif target_hint == "auto":
        forced_target = None
    else:
        raise ValueError("device_target must be auto, Ascend/NPU, or CPU")

    if value.lower() == "cpu":
        return "CPU", 0
    if value.lower().startswith("cuda"):
        raise ValueError("Ascend does not use CUDA device strings; use npu:0 or cpu")

    match = re.fullmatch(r"(?i)(ascend|npu|cpu|auto)(?::(\d+))?", value)
    if match:
        family, device_id = match.groups()
        family = family.lower()
        if family == "cpu":
            return "CPU", 0
        if family in {"ascend", "npu"}:
            return "Ascend", int(device_id or 0)
        target = forced_target or _auto_target()
        return target, int(device_id or 0) if target == "Ascend" else 0

    if value.isdigit():
        target = forced_target or _auto_target()
        return target, int(value) if target == "Ascend" else 0
    raise ValueError(f"Unrecognised device specification: {device!r}")


def setup_context(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialise MindSpore and return effective runtime metadata."""
    import mindspore as ms

    target, device_id = parse_device_spec(
        str(config.get("device", "npu:0")),
        str(config.get("device_target", "auto")),
    )
    if target != "Ascend":
        device_id = 0
    mode = ms.GRAPH_MODE if config.get("ms_mode") == "graph" else ms.PYNATIVE_MODE

    set_device = getattr(ms, "set_device", None)
    if callable(set_device):
        set_device(target, device_id)
        ms.set_context(mode=mode)
    else:
        context_kwargs: Dict[str, Any] = {"mode": mode, "device_target": target}
        if target == "Ascend":
            context_kwargs["device_id"] = device_id
        ms.set_context(**context_kwargs)
    ms.set_seed(int(config.get("seed", 1993)))

    if mode == ms.GRAPH_MODE:
        logging.info("MindSpore GRAPH_MODE selected for the dynamic prompt-pool trainer.")
        if target == "Ascend":
            ms.set_context(jit_config={"jit_level": "O0"})

    return {
        "mindspore_version": getattr(ms, "__version__", "unknown"),
        "device_target": target,
        "device_id": device_id,
        "mode": "graph" if mode == ms.GRAPH_MODE else "pynative",
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH", ""),
    }


def synchronize() -> None:
    """Best-effort device synchronisation for timing and diagnostics."""
    try:
        import mindspore as ms

        target = str(ms.get_context("device_target")).strip().lower()
        if target not in {"ascend", "npu"}:
            return
        runtime = getattr(ms, "runtime", None)
        if runtime is not None and hasattr(runtime, "synchronize"):
            runtime.synchronize()
            return
        from mindspore import hal

        hal.synchronize()
    except (ImportError, AttributeError, RuntimeError):
        pass


def describe_device(info: Dict[str, Any]) -> str:
    return "MindSpore {} | target={} | id={} | mode={}".format(
        info["mindspore_version"],
        info["device_target"],
        info["device_id"],
        info["mode"],
    )
