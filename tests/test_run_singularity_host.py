from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


COLAB = Path(__file__).parents[1] / "colab"
sys.path.insert(0, str(COLAB))  # the runner imports run_dance_swap the way it does on Colab
SPEC = importlib.util.spec_from_file_location("run_singularity_host", COLAB / "run_singularity_host.py")
assert SPEC and SPEC.loader
run_singularity_host = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_singularity_host
SPEC.loader.exec_module(run_singularity_host)


def canvas() -> dict:
    """The nodes the runner fills, shaped like the host canvas built from the RunningHub workflow."""
    combine = {"frame_rate": 24, "filename_prefix": "minimaxh3", "format": "video/h264-mp4"}
    return {"nodes": [
        {"id": 214, "type": "LoadImage", "mode": 0, "widgets_values": ["old.png", "image"]},
        {"id": 304, "type": "LoadAudio", "mode": 0, "widgets_values": ["dub.wav", None]},
        {"id": 2, "type": "PrimitiveStringMultiline", "mode": 0, "widgets_values": ["old prompt"]},
        {"id": 17, "type": "PrimitiveFloat", "mode": 0, "widgets_values": [10]},
        {"id": 20, "type": "ResolutionSelector", "mode": 0, "widgets_values": ["16:9 (Widescreen)", 1.5, 32]},
        {"id": 233, "type": "SelfLiftH3Sampler", "mode": 0,
         "widgets_values": [5, "fixed", 1, 4, 0.4, 0, 0.5, 1, "upscaler.safetensors", False]},
        {"id": 243, "type": "VHS_VideoCombine", "mode": 0, "widgets_values": dict(combine)},
        {"id": 270, "type": "VHS_VideoCombine", "mode": 0, "widgets_values": dict(combine)},
    ]}


class FillCanvasTests(unittest.TestCase):
    def fill(self) -> dict:
        filled = run_singularity_host.fill_canvas(
            canvas(), image="host_portrait.png", audio="host_dub.wav", prompt="new prompt", seconds=14.25,
            seed=101, megapixels=2.0, prefix="host-singularity/host")
        return {node["id"]: node for node in filled["nodes"]}

    def test_inputs_land_on_their_nodes(self) -> None:
        nodes = self.fill()
        self.assertEqual(nodes[214]["widgets_values"], ["host_portrait.png", "image"])
        self.assertEqual(nodes[304]["widgets_values"][0], "host_dub.wav")
        self.assertEqual(nodes[2]["widgets_values"], ["new prompt"])
        self.assertEqual(nodes[20]["widgets_values"], ["16:9 (Widescreen)", 2.0, 32])

    def test_length_is_the_next_whole_second_after_the_dub(self) -> None:
        self.assertEqual(self.fill()[17]["widgets_values"], [15.0])

    def test_seed_is_set_and_the_rest_of_the_sampler_kept(self) -> None:
        values = self.fill()[233]["widgets_values"]
        self.assertEqual(values[0], 101)
        self.assertEqual(values[1:], [ "fixed", 1, 4, 0.4, 0, 0.5, 1, "upscaler.safetensors", False])

    def test_each_saved_clip_gets_its_own_prefix(self) -> None:
        nodes = self.fill()
        self.assertEqual(nodes[243]["widgets_values"]["filename_prefix"], "host-singularity/host_raw")
        self.assertEqual(nodes[270]["widgets_values"]["filename_prefix"], "host-singularity/host_face")


class WindowTests(unittest.TestCase):
    def test_whole_dub_by_default(self) -> None:
        self.assertAlmostEqual(run_singularity_host.window_seconds(14.25, 0.0, None), 14.25)

    def test_a_sentence_window(self) -> None:
        self.assertAlmostEqual(run_singularity_host.window_seconds(14.25, 4.86, 5.2), 5.2)

    def test_window_is_clipped_to_the_dub(self) -> None:
        self.assertAlmostEqual(run_singularity_host.window_seconds(14.25, 10.08, 9.0), 4.17)


if __name__ == "__main__":
    unittest.main()
