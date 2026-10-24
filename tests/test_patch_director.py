from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "colab" / "patch_director.py"
SPEC = importlib.util.spec_from_file_location("patch_director", MODULE_PATH)
assert SPEC and SPEC.loader
patch_director = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = patch_director
SPEC.loader.exec_module(patch_director)

CALL = ("        def _condition(condition_prompt, condition_width, condition_height):\n"
        + patch_director.OLD
        + "                ref_image_size=ref_image_size, ref_images=ref_images, ref_audios=ref_audios,\n"
        "                ref_videos=ref_videos, ref_video_audios=ref_video_audios)\n")


class PatchDirectorTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        folder = Path(tempfile.mkdtemp())
        path = folder / "studio_node.py"
        path.write_text(text, encoding="utf-8")
        return path

    def test_positional_call_becomes_keyword_call(self) -> None:
        path = self.write(CALL)
        self.assertEqual(patch_director.patch(path), "patched")
        text = path.read_text(encoding="utf-8")
        self.assertIn("prompt=condition_prompt", text)
        self.assertNotIn("clip, vae, audio_vae, condition_prompt", text)
        compile(text.replace("        def", "def", 1).replace("\n        ", "\n    "), "studio", "exec")

    def test_second_run_is_a_no_op(self) -> None:
        path = self.write(CALL)
        patch_director.patch(path)
        self.assertEqual(patch_director.patch(path), "already patched")

    def test_unknown_source_is_left_alone(self) -> None:
        path = self.write("print('something else')\n")
        with self.assertRaises(SystemExit):
            patch_director.patch(path)
        self.assertEqual(path.read_text(encoding="utf-8"), "print('something else')\n")


if __name__ == "__main__":
    unittest.main()
