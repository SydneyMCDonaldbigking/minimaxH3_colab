#!/usr/bin/env python3
"""Upscale a finished clip to delivery size with the private pipeline's own upscale step.

The runtime restored by bootstrap_colab.py carries comfyui_quick_workflow, whose
scripts/cc_upscale.py does what the 5090 server does after every job: RealESRGAN x2
through the Director's H3DirectorVideoUpscale node (audio, frame rate and duration
kept), an ffmpeg pass to the delivery size, then the machine QA.  This wrapper only
gives it a delivery size that keeps the clip's own aspect, runs it, and copies the
result to MyDrive/minimaxH3_colab/deliveries/upscale/<job-id>/.

Run it once on the finished clip, never between swap passes: every swap pass
shrinks its source back to about one megapixel, so an earlier upscale is wasted.

--dry-run builds the same API prompt cc_upscale sends and validates it on CPU.
--cpu-test runs the whole chain on CPU on a tiny cut (6 frames, 320 px wide) to prove
the flow before a GPU session; its result is not a delivery.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run_dance_swap import has_gpu, probe, restart_comfy, run

HERE = Path(__file__).resolve().parent
COMFY = Path("/content/ComfyUI")
SOURCE = Path("/content/comfyui_quick_workflow")
RUNTIME_PYTHON = Path("/content/h3env/bin/python")
TEST_SECONDS, TEST_WIDTH = 0.25, 320


def delivery_size(width: int, height: int, target_height: int) -> tuple[int, int]:
    """Width and height at target_height with the source aspect kept (even width for x264)."""
    return round(target_height * width / height / 2) * 2, target_height


def job_config(base: dict, width: int, height: int) -> dict:
    """The Colab production config with this clip's delivery size and a QA that checks it."""
    config = json.loads(json.dumps(base))
    config["upscale"].update({"delivery_width": width, "delivery_height": height})
    # The production QA minimum is the vertical 1440x2560 spec; here the clip's own size is the spec.
    config.setdefault("qa", {}).update({"min_width": width, "min_height": height})
    return config


PROMPT_SNIPPET = """
import json, sys
sys.path.insert(0, sys.argv[1])
from src.config import load_config
from src.workflow import build_upscale_prompt
config = load_config(sys.argv[2])
prompt = build_upscale_prompt(sys.argv[3], config["upscale"], sys.argv[4])
open(sys.argv[5], "w", encoding="utf-8").write(json.dumps(prompt, ensure_ascii=False, indent=2))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="成片超分：RealESRGAN x2 再转到交付尺寸，保留原音轨")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/minimaxH3_colab"))
    parser.add_argument("--video", type=Path, required=True, help="要超分的成片（所有换人遍都跑完之后的那条）")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--height", type=int, default=1080, help="交付高度，宽度按原比例算；默认 1080")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="CPU 检查：只生成并校验超分的 API prompt")
    mode.add_argument("--cpu-test", action="store_true",
                      help="CPU 流程测试：截 6 帧缩到 320 宽，在 CPU 上把整条链路走一遍")
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"找不到文件：{args.video}")
    gpu = has_gpu()
    if not (gpu or args.dry_run or args.cpu_test):
        raise SystemExit("当前没有 GPU；只校验请加 --dry-run，CPU 上走一遍流程请加 --cpu-test")

    run([sys.executable, HERE / "bootstrap_colab.py", "--drive-root", args.drive_root,
         "--profile", "dance_swap", "--skip-models", *([] if gpu else ["--prepare-on-cpu"])])
    import yaml  # Colab's system Python has PyYAML; bootstrap_colab.py relies on it too

    base = yaml.safe_load((SOURCE / "config" / "production.colab.yaml").read_text(encoding="utf-8"))
    model = args.drive_root / "models" / "upscale_models" / base["upscale"]["model"]
    if not model.is_file():
        raise SystemExit(f"Drive 里没有超分模型：{model}")

    work = Path("/content/h3-upscale") / args.job_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    clip = work / "source.mp4"
    if args.cpu_test:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video, "-t", TEST_SECONDS, "-r", "24",
             "-vf", f"scale={TEST_WIDTH}:-2", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", clip])
    else:
        shutil.copy2(args.video, clip)
    _, width, height = probe(clip)
    target_height = height * 2 if args.cpu_test else args.height
    out_width, out_height = delivery_size(width, height, target_height)

    config_path = SOURCE / "config" / f"upscale.{args.job_id}.yaml"
    config_path.write_text(yaml.safe_dump(job_config(base, out_width, out_height), allow_unicode=True,
                                          sort_keys=False), encoding="utf-8")
    summary = {"job_id": args.job_id, "source": str(args.video), "source_size": [width, height],
               "delivery_size": [out_width, out_height], "model": model.name,
               "mode": "dry-run" if args.dry_run else "cpu-test" if args.cpu_test else "gpu"}

    if args.dry_run:
        # cc_upscale uploads the clip to input/h3upscale/<job-id>/ first; stage it there by hand.
        staged = COMFY / "input" / "h3upscale" / args.job_id / clip.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip, staged)
        api_path, report_path = work / "api_prompt.json", work / "validation.json"
        run([RUNTIME_PYTHON, "-c", PROMPT_SNIPPET, SOURCE, config_path,
             f"h3upscale/{args.job_id}/{clip.name}", args.job_id, api_path])
        run([RUNTIME_PYTHON, HERE / "validate_api_prompt.py", api_path, report_path], cwd=COMFY)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary["validation_ok"] = report["all_ok"]
        if not report["all_ok"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if report["all_ok"] else 1

    restart_comfy(args.drive_root, cpu=not gpu)
    started = time.time()
    out = work / "out"
    run([RUNTIME_PYTHON, SOURCE / "scripts" / "cc_upscale.py", "--config", config_path,
         "--input", clip, "--job-id", args.job_id, "--out-dir", out], cwd=SOURCE)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    summary.update({"elapsed_seconds": round(time.time() - started, 1),
                    "qa_passed": report["qa"]["passed"], "qa_failures": report["qa"]["failures"]})

    if args.cpu_test:
        # Not a delivery: keep it out of deliveries/, but leave the report where Drive syncs it.
        dest = args.drive_root / "logs" / "upscale-cputest" / args.job_id
    else:
        dest = args.drive_root / "deliveries" / "upscale" / args.job_id
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{args.job_id}_{out_height}p.mp4"
    shutil.copy2(out / f"{args.job_id}.mp4", dest / name)
    shutil.copy2(out / "report.json", dest / "report.json")
    summary.update({"file": str(dest / name)})
    (dest / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
