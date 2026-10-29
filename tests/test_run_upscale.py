from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


COLAB = Path(__file__).parents[1] / "colab"
sys.path.insert(0, str(COLAB))  # run_upscale imports run_dance_swap the way it does on Colab
SPEC = importlib.util.spec_from_file_location("run_upscale", COLAB / "run_upscale.py")
assert SPEC and SPEC.loader
run_upscale = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_upscale
SPEC.loader.exec_module(run_upscale)


class DeliverySizeTests(unittest.TestCase):
    def test_swap_outputs_keep_their_aspect(self) -> None:
        self.assertEqual(run_upscale.delivery_size(1344, 768, 1080), (1890, 1080))  # 16:9 canvas
        self.assertEqual(run_upscale.delivery_size(1184, 896, 1080), (1428, 1080))  # 4:3 canvas

    def test_width_is_always_even(self) -> None:
        for width, height in ((1344, 768), (1184, 896), (320, 182), (1000, 777)):
            out_width, _ = run_upscale.delivery_size(width, height, 1080)
            self.assertEqual(out_width % 2, 0)

    def test_cpu_test_clip_is_doubled(self) -> None:
        self.assertEqual(run_upscale.delivery_size(320, 182, 364), (640, 364))


class JobConfigTests(unittest.TestCase):
    def base(self) -> dict:
        return {"upscale": {"model": "RealESRGAN_x2plus.pth", "scale": 2.0, "frame_batch_size": 1,
                            "delivery_width": 1440, "delivery_height": 2560},
                "qa": {"min_width": 1400, "min_height": 2500, "require_audio": True}}

    def test_delivery_size_and_qa_follow_the_clip(self) -> None:
        config = run_upscale.job_config(self.base(), 1890, 1080)
        self.assertEqual((config["upscale"]["delivery_width"], config["upscale"]["delivery_height"]),
                         (1890, 1080))
        self.assertEqual((config["qa"]["min_width"], config["qa"]["min_height"]), (1890, 1080))
        self.assertEqual(config["upscale"]["frame_batch_size"], 1)  # Colab's RAM-safe batch is kept
        self.assertTrue(config["qa"]["require_audio"])

    def test_base_config_is_not_changed(self) -> None:
        base = self.base()
        run_upscale.job_config(base, 1890, 1080)
        self.assertEqual(base, self.base())


if __name__ == "__main__":
    unittest.main()
