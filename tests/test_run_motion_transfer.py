from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "colab" / "run_motion_transfer.py"
SPEC = importlib.util.spec_from_file_location("run_motion_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
run_motion_transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_motion_transfer
SPEC.loader.exec_module(run_motion_transfer)

ACTION_LINE = ("<Video 1>: reference - use only the subject motion, body mechanics and blocking "
               "from <Video 1>; do not copy identities, costumes, locations or visual style from that video.")


class PromptCheckTests(unittest.TestCase):
    def test_six_part_prompt_without_mode_sentence_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            run_motion_transfer.check_prompt("retention_analysis:\n<Video 1> appearance: not_preserved.", "action")

    def test_six_part_prompt_with_mode_sentence_passes(self) -> None:
        run_motion_transfer.check_prompt(f"retention_analysis:\n{ACTION_LINE}", "action")

    def test_sentence_must_match_the_chosen_mode(self) -> None:
        with self.assertRaises(SystemExit):
            run_motion_transfer.check_prompt(f"retention_analysis:\n{ACTION_LINE}", "camera")

    def test_prompt_without_retention_analysis_is_left_to_the_node(self) -> None:
        run_motion_transfer.check_prompt("A dancer from <Picture 1> follows <Video 1>.", "action")


class SegmentTests(unittest.TestCase):
    def test_frames_snap_down_to_the_h3_grid(self) -> None:
        self.assertEqual(run_motion_transfer.aligned_frames(8.0), 192)  # 8 s lands on the grid
        self.assertEqual(run_motion_transfer.aligned_frames(9.535), 226)
        self.assertEqual((run_motion_transfer.aligned_frames(5.0) - 5) % 17, 0)

    def test_long_clip_splits_into_equal_aligned_segments(self) -> None:
        count, seconds = run_motion_transfer.plan_segments(19.07, None, 0)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(seconds, 226 / 24)
        self.assertLessEqual(count * seconds, 19.07)

    def test_short_clip_stays_one_segment(self) -> None:
        self.assertEqual(run_motion_transfer.plan_segments(8.0, None, 0), (1, 8.0))

    def test_too_long_segments_are_refused(self) -> None:
        with self.assertRaises(SystemExit):
            run_motion_transfer.plan_segments(40.0, None, 2)


class SeedTests(unittest.TestCase):
    def test_explicit_seeds_keep_their_order(self) -> None:
        self.assertEqual(run_motion_transfer.parse_seeds("101, 7,202", 5), [101, 7, 202])

    def test_draws_give_that_many_distinct_random_seeds(self) -> None:
        seeds = run_motion_transfer.parse_seeds(None, 3)
        self.assertEqual(len(set(seeds)), 3)

    def test_at_least_one_draw(self) -> None:
        self.assertEqual(len(run_motion_transfer.parse_seeds(None, 0)), 1)


class LogCheckTests(unittest.TestCase):
    def check(self, lines: list[str], want_audio: bool = True) -> list[str]:
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "comfyui.log"
            log.write_text("earlier job 段1 开始生成（参考图 0 张）\n", encoding="utf-8")
            offset = log.stat().st_size
            with log.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            return run_motion_transfer.check_log(log, offset, want_audio)

    def test_good_run_has_no_problems(self) -> None:
        self.assertEqual(self.check([
            "[H3导演台] ==== 段1 开始生成（参考图 1 张）====",
            "[H3导演台] 段1 参考视频 <Video 1>: motion_small.mp4（240 帧，已发送原音轨）",
        ]), [])

    def test_lines_before_the_offset_are_ignored(self) -> None:
        self.assertEqual(len(self.check([])), 2)

    def test_missing_picture_and_audio_are_reported(self) -> None:
        problems = self.check([
            "[H3导演台] ==== 段1 开始生成（参考图 0 张）====",
            "[H3导演台] 段1 参考视频 <Video 1>: motion_small.mp4（240 帧，仅参考画面、不发送原音轨）",
        ])
        self.assertEqual(len(problems), 2)

    def test_picture_only_audio_is_fine_when_audio_is_off(self) -> None:
        self.assertEqual(self.check([
            "[H3导演台] ==== 段1 开始生成（参考图 1 张）====",
            "[H3导演台] 段1 参考视频 <Video 1>: motion_small.mp4（240 帧，仅参考画面、不发送原音轨）",
        ], want_audio=False), [])


if __name__ == "__main__":
    unittest.main()
