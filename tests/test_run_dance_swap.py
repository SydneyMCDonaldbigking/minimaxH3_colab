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
        {"id": 53, "type": "SAM3_VideoTrack", "mode": 0, "widgets_values": [0.5, 1, 1]},
        {"id": 55, "type": "SAM3_TrackToMask", "mode": 0, "pos": [100, 200],
         "outputs": [{"name": "masks", "type": "MASK", "links": [70]}]},
        {"id": 56, "type": "MVEx_MaskCleanup", "mode": 0,
         "inputs": [{"name": "mask", "type": "MASK", "link": 70}],
         "outputs": [{"name": "mask", "type": "MASK", "links": [71]}]},
        {"id": 123, "type": "MVEx_SubjectCrop", "mode": 0,
         "inputs": [{"name": "images", "type": "IMAGE", "link": None},
                    {"name": "masks", "type": "MASK", "link": 71}]},
    ], "links": [[70, 55, 0, 56, 0, "MASK"], [71, 56, 0, 123, 1, "MASK"]],
       "last_node_id": 126, "last_link_id": 71}


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
        self.assertEqual(nodes[53]["widgets_values"], [0.5, 4, 1])  # threshold and interval untouched

    def test_sam_object_limit_can_be_set(self) -> None:
        filled = run_dance_swap.fill_canvas(
            canvas(), video="v", image="i", target="t", prompt="p", seconds=5, start_frames=0,
            aspect="16:9", sam_max_objects=1)
        self.assertEqual(self.by_id(filled)[53]["widgets_values"][1], 1)

    def test_bypassed_scene_image_is_left_alone(self) -> None:
        self.assertEqual(self.by_id(self.fill())[126]["widgets_values"][0], "scene_reference.png")

    def test_person_mask_leaves_the_wiring_alone(self) -> None:
        filled = self.fill()
        self.assertEqual(filled["links"], canvas()["links"])
        self.assertNotIn("InvertMask", [node["type"] for node in filled["nodes"]])

    def test_background_mask_inverts_before_cleanup(self) -> None:
        filled = run_dance_swap.fill_canvas(
            canvas(), video="v", image="scene.png", target="person", prompt="p", seconds=5,
            start_frames=0, aspect="16:9", mask="background")
        nodes = self.by_id(filled)
        invert = next(node for node in filled["nodes"] if node["type"] == "InvertMask")
        self.assertEqual(invert["id"], 127)  # above every existing id
        links = {link[0]: link for link in filled["links"]}
        into_invert = links[invert["inputs"][0]["link"]]
        self.assertEqual(into_invert[1:5], [55, 0, 127, 0])
        into_cleanup = links[nodes[56]["inputs"][0]["link"]]
        self.assertEqual(into_cleanup[1:6], [127, 0, 56, 0, "MASK"])
        self.assertEqual(invert["outputs"][0]["links"], [into_cleanup[0]])
        self.assertEqual(links[71][1:5], [56, 0, 123, 1])  # cleanup still feeds the crop
        self.assertEqual((filled["last_node_id"], filled["last_link_id"]), (127, 72))

    def test_unexpected_canvas_is_refused(self) -> None:
        broken = canvas()
        broken["nodes"].append({"id": 99, "type": "CLIPTextEncode", "mode": 0, "widgets_values": ["x"]})
        with self.assertRaises(SystemExit):
            run_dance_swap.fill_canvas(broken, video="v", image="i", target="t", prompt="p",
                                       seconds=5, start_frames=0, aspect="16:9")


class AspectTests(unittest.TestCase):
    def test_common_sources_map_to_their_canvas_aspect(self) -> None:
        self.assertEqual(run_dance_swap.nearest_aspect(960, 720), "4:3")
        self.assertEqual(run_dance_swap.nearest_aspect(1280, 720), "16:9")
        self.assertEqual(run_dance_swap.nearest_aspect(720, 1280), "9:16")
        self.assertEqual(run_dance_swap.nearest_aspect(1080, 1080), "1:1")

    def test_odd_sizes_pick_the_closest(self) -> None:
        self.assertEqual(run_dance_swap.nearest_aspect(1920, 1040), "16:9")
        self.assertEqual(run_dance_swap.nearest_aspect(2560, 1080), "21:9")


class DurationTests(unittest.TestCase):
    def test_full_length_clip_never_asks_for_more_frames_than_it_has(self) -> None:
        frames = run_dance_swap.aligned_frames(15.03)
        self.assertEqual(frames, 345)
        self.assertLessEqual(frames, 15.03 * 24)
        self.assertEqual((frames - 5) % 17, 0)

    def test_grid_durations_are_kept(self) -> None:
        self.assertEqual(run_dance_swap.aligned_frames(8.0), 192)


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
