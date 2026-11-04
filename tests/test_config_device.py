from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ldeprompt_mindspore.config import load_config, resolve_seeds
from ldeprompt_mindspore.device import describe_device, parse_device_spec, setup_context


class ConfigTests(unittest.TestCase):
    def test_load_config_merges_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"dataset": "synthetic", "top_k": 2}), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config["dataset"], "synthetic")
        self.assertEqual(config["top_k"], 2)
        self.assertEqual(config["pool_size"], 10)

    def test_invalid_top_k_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"pool_size": 2, "top_k": 3}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top_k"):
                load_config(path)

    def test_seed_resolution(self):
        self.assertEqual(resolve_seeds({"seed": [1, 3]}), [1, 3])
        self.assertEqual(resolve_seeds({"seed": 7}), [7])
        self.assertEqual(resolve_seeds({"seed": [1, 3]}, override=9), [9])


class DeviceTests(unittest.TestCase):
    def test_device_aliases(self):
        self.assertEqual(parse_device_spec("cpu"), ("CPU", 0))
        self.assertEqual(parse_device_spec("npu:3"), ("Ascend", 3))
        self.assertEqual(parse_device_spec("Ascend:2"), ("Ascend", 2))
        self.assertEqual(parse_device_spec("1", "ascend"), ("Ascend", 1))

    def test_cuda_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            parse_device_spec("cuda:0")

    def test_cpu_context(self):
        info = setup_context(
            {"device": "cpu", "device_target": "auto", "ms_mode": "pynative", "seed": 1}
        )
        self.assertEqual(info["device_target"], "CPU")
        self.assertEqual(info["device_id"], 0)
        self.assertIn("target=CPU", describe_device(info))


if __name__ == "__main__":
    unittest.main()
