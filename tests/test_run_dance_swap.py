from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "colab" / "run_dance_swap.py"
SPEC = importlib.util.spec_from_file_location("run_dance_swap", MODULE_PATH)
assert SPEC and SPEC.loader
run_dance_swap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_dance_swap
SPEC.loader.exec_module(run_dance_swap)


def canvas() -> dict:
    """The nodes run_dance_swap fills, shaped like the FastH3 character-swap canvas."""
    return {"nodes": [
        {"id": 43, "type": "VHS_LoadVideo", "mode": 0,
         "widgets_values": {"video": "source_dance.mp4", "skip_first_frames": 0, "force_rate": 24}},
        {"id": 9, "type": "LoadImage", "mode": 0, "widgets_values": ["character_reference.png", "image"]},
        {"id": 126, "type": "LoadImage", "mode": 4, "widgets_values": ["scene_reference.png", "image"]},
        {"id": 50, "type": "CLIPTextEncode", "mode": 0, "widgets_values": ["the dancer on the right"]},
        {"id": 2, "type": "PrimitiveStringMultiline", "mode": 0, "widgets_values": ["subject_definitions: ..."]},
        {"id": 17, "type": "PrimitiveFloat", "mode": 0, "widgets_values": [10]},
        {"id": 44, "type": "PrimitiveNode", "mode": 0, "widgets_values": ["16:9 (Widescreen)", "fixed", ""]},
    ]}


class FillCanvasTests(unittest.TestCase):
    def fill(self) -> dict:
        return run_dance_swap.fill_canvas(
            canvas(), video="job_source.mp4", image="job_character.png", target="the girl in red",
            prompt="new prompt", seconds=8.004, start_frames=48, aspect="9:16")

    def by_id(self, filled: dict) -> dict:
        return {node["id"]: node for node in filled["nodes"]}

    def test_inputs_land_on_the_right_nodes(self) -> None:
        nodes = self.by_id(self.fill())
        self.assertEqual(nodes[43]["widgets_values"]["video"], "job_source.mp4")
        self.assertEqual(nodes[43]["widgets_values"]["skip_first_frames"], 48)
        self.assertEqual(nodes[43]["widgets_values"]["force_rate"], 24)
        self.assertEqual(nodes[9]["widgets_values"], ["job_character.png", "image"])
        self.assertEqual(nodes[50]["widgets_values"], ["the girl in red"])
        self.assertEqual(nodes[2]["widgets_values"], ["new prompt"])
        self.assertEqual(nodes[17]["widgets_values"], [8.0])
        self.assertEqual(nodes[44]["widgets_values"][0], "9:16")

    def test_bypassed_scene_image_is_left_alone(self) -> None:
        self.assertEqual(self.by_id(self.fill())[126]["widgets_values"][0], "scene_reference.png")

    def test_unexpected_canvas_is_refused(self) -> None:
        broken = canvas()
        broken["nodes"].append({"id": 99, "type": "CLIPTextEncode", "mode": 0, "widgets_values": ["x"]})
        with self.assertRaises(SystemExit):
            run_dance_swap.fill_canvas(broken, video="v", image="i", target="t", prompt="p",
                                       seconds=5, start_frames=0, aspect="16:9")


class ApiNodeTests(unittest.TestCase):
    def test_finds_the_single_node_of_a_type(self) -> None:
        prompt = {"31": {"class_type": "RandomNoise"}, "42": {"class_type": "VHS_VideoCombine"}}
        self.assertEqual(run_dance_swap.api_node(prompt, "RandomNoise"), "31")

    def test_refuses_ambiguous_types(self) -> None:
        prompt = {"1": {"class_type": "RandomNoise"}, "2": {"class_type": "RandomNoise"}}
        with self.assertRaises(SystemExit):
            run_dance_swap.api_node(prompt, "RandomNoise")


if __name__ == "__main__":
    unittest.main()
