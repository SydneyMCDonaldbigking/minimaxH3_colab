from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "colab" / "bootstrap_colab.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_colab", MODULE_PATH)
assert SPEC and SPEC.loader
bootstrap_colab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap_colab
SPEC.loader.exec_module(bootstrap_colab)


class BootstrapColabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = bootstrap_colab.Layout(Path("/drive"), Path("/content"))

    def test_snapshot_key_tracks_cuda_13_stack(self) -> None:
        self.assertIn("torch2110-cu130", self.layout.snapshot_key)
        self.assertIn("comfybc5f880", self.layout.snapshot_key)
        self.assertTrue(self.layout.snapshot_key.endswith("-s3"))

    def test_dance_swap_model_folders_include_sam_checkpoints(self) -> None:
        self.assertIn("checkpoints", bootstrap_colab.MODEL_FOLDERS)

    def test_dance_swap_preview_decoder_folder_is_linked(self) -> None:
        # ModelPreviewOverrideKJ lists tiny_vae from vae_approx; taeh3 missing there rejects the video output.
        self.assertIn("vae_approx", bootstrap_colab.MODEL_FOLDERS)

    def test_dance_swap_plugins_are_pinned(self) -> None:
        names = {name for name, _url, _ref in bootstrap_colab.CUSTOM_NODE_REPOS}
        self.assertEqual(
            names,
            {"ComfyUI-KJNodes", "ComfyUI-VideoHelperSuite", "MaskVidExperiments"},
        )
        for _name, _url, ref in bootstrap_colab.CUSTOM_NODE_REPOS:
            self.assertRegex(ref, r"^[0-9a-f]{40}$")

    def test_blackwell_skips_generic_sageattention(self) -> None:
        capability = SimpleNamespace(stdout="[12, 0]\n", returncode=0)
        with patch.object(bootstrap_colab.subprocess, "run", return_value=capability) as run_mock:
            self.assertFalse(bootstrap_colab.ensure_sage_attention(self.layout, {}))
        self.assertEqual(run_mock.call_count, 1)

    def test_sageattention_install_failure_is_nonfatal(self) -> None:
        capability = SimpleNamespace(stdout="[9, 0]\n", returncode=0)
        missing = SimpleNamespace(stdout="", returncode=1)
        with (
            patch.object(bootstrap_colab.subprocess, "run", side_effect=(capability, missing)),
            patch.object(
                bootstrap_colab,
                "run",
                side_effect=subprocess.CalledProcessError(1, ["pip", "install"]),
            ),
        ):
            self.assertFalse(bootstrap_colab.ensure_sage_attention(self.layout, {}))


if __name__ == "__main__":
    unittest.main()
