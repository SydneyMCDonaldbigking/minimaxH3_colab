#!/usr/bin/env python3
"""Run one H3 motion-transfer job end to end on a Colab runtime.

Appearance comes from --image (the character, and the scene it stands in), motion
from --video (the dance). The whole frame is regenerated, unlike the dance_swap
canvas, which only repaints one person inside the original video.

Steps: restore the runtime from the Drive snapshot (ref2va profile), start
ComfyUI without the queue worker, shrink the reference video, run
colab/motion_transfer.py, check the ComfyUI log lines that prove the picture,
video and audio references reached the model, then save everything under
MyDrive/minimaxH3_colab/deliveries/motion/<job-id>/.

--seeds/--draws give one clip per seed from the same inputs, so routes and
settings can be compared over a spread of draws rather than a single lucky one.

Drive must already be mounted at /content/drive (`colab drivemount`, or
google.colab.drive.mount in a notebook).  --dry-run runs on CPU: it restores the
runtime without models, prepares the video and writes the API prompt only.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME_PYTHON = Path("/content/h3env/bin/python")
SOURCE_ROOT = Path("/content/comfyui_quick_workflow")
MAX_SECONDS = 15.0  # Ref2VA caps both a segment and its reference video at 15 s
# The node skips its own video declaration when the prompt has a retention_analysis
# and a <Video tag, so the official sentence for the chosen mode must be in the prompt.
MODE_PHRASES = {
    "action": "the subject motion, body mechanics and blocking",
    "camera": "the camera trajectory, framing changes and camera orientation",
    "rhythm": "the visible action timing, acceleration, pauses and pacing",
    "comprehensive": "the camera trajectory, subject motion, blocking and visible pacing",
}
SIZES = {  # output, reference video (long edge 960 keeps choreography without blowing VRAM)
    "portrait": ((768, 1344), (540, 960)),
    "landscape": ((1344, 768), (960, 540)),
}


def run(cmd: list, **kwargs) -> None:
    cmd = [str(part) for part in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def probe(video: Path) -> tuple[float, int, int]:
    """Duration in seconds and display width/height of the first video stream."""
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout)
    stream = info["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = abs(int(next((d.get("rotation", 0) for d in stream.get("side_data_list", [])), 0)))
    if rotation in (90, 270):  # phone clips are often stored landscape with a rotation flag
        width, height = height, width
    return float(info["format"]["duration"]), width, height


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode == 0


def check_prompt(text: str, mode: str) -> None:
    if "retention_analysis" in text and "<Video" in text and MODE_PHRASES[mode] not in text:
        raise SystemExit(
            f"提示词写了 retention_analysis，但没有 {mode} 模式的官方声明原句"
            f"（需包含 “{MODE_PHRASES[mode]}”）；不补这句 --video-mode 不会进模型。")


def parse_seeds(seeds: str | None, draws: int) -> list[int]:
    if seeds:
        return [int(seed) for seed in seeds.split(",") if seed.strip()]
    return [secrets.randbelow(10**15) for _ in range(max(1, draws))]


def host_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def restart_comfy(drive_root: Path) -> None:
    """Start a fresh ComfyUI so models cached by an earlier job do not pile up in host RAM."""
    pattern = "main.py --listen 127.0.0.1"
    subprocess.run(["pkill", "-f", pattern], check=False)
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode:
            break
        time.sleep(1)
    run([sys.executable, HERE / "launch_colab.py", "--drive-root", drive_root, "--no-worker"])


def check_log(log: Path, offset: int, want_audio: bool) -> list[str]:
    """Return the problems found in the ComfyUI log lines written during this job."""
    with log.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode("utf-8", "replace")
    starts = [line for line in text.splitlines() if "开始生成" in line]
    videos = [line for line in text.splitlines() if "参考视频 <Video" in line]
    for line in starts + videos:
        print("  log:", line.strip())
    problems = []
    if not starts:
        problems.append("日志里没有“段1 开始生成”")
    elif any("参考图 0 张" in line for line in starts):
        problems.append("参考图 0 张：角色图没挂上")
    if not videos:
        problems.append("没有“参考视频 <Video 1>”：video_refs 没生效")
    elif want_audio and not all("已发送原音轨" in line for line in videos):
        problems.append("参考视频只送了画面，音轨没发送")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="H3 动作迁移：角色图定外观和场景，舞蹈视频定动作")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/minimaxH3_colab"))
    parser.add_argument("--image", type=Path, action="append", required=True, help="角色图，可重复")
    parser.add_argument("--video", type=Path, required=True, help="舞蹈参考视频（单个连续镜头）")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="从参考视频第几秒开始取")
    parser.add_argument("--duration", type=float, help="成片秒数，默认取参考视频长度，最多 15")
    parser.add_argument("--orientation", choices=["auto", *sorted(SIZES)], default="auto",
                        help="成片横竖屏，默认跟参考视频一致")
    parser.add_argument("--video-mode", choices=sorted(MODE_PHRASES), default="action")
    parser.add_argument("--no-video-audio", action="store_true", help="不把参考视频的音轨送进模型")
    parser.add_argument("--mux-original-audio", action="store_true",
                        help="另存一份用参考视频原音轨整条覆盖的成片")
    parser.add_argument("--seeds", help="逗号分隔的 seed，例如 101,102,103；每个 seed 出一条")
    parser.add_argument("--draws", type=int, default=1, help="不给 --seeds 时随机抽几条")
    parser.add_argument("--dry-run", action="store_true", help="CPU 检查：只生成 API prompt，不生成视频")
    args = parser.parse_args()

    prompt_text = args.prompt_file.read_text(encoding="utf-8")
    check_prompt(prompt_text, args.video_mode)
    for path in [*args.image, args.video]:
        if not path.is_file():
            raise SystemExit(f"找不到文件：{path}")
    if not args.dry_run and not has_gpu():
        raise SystemExit("当前没有 GPU；只做检查请加 --dry-run")

    work = Path("/content/h3-motion") / args.job_id
    work.mkdir(parents=True, exist_ok=True)
    # A dry run is a check, not a delivery: keep its files on the runtime.
    out_dir = work / "check" if args.dry_run else args.drive_root / "deliveries" / "motion" / args.job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --skip-models: the ref2va files are already in Drive, and ComfyUI rejects the prompt if
    # one is missing; checking them here would only spend GPU minutes re-reading Drive.
    bootstrap = [sys.executable, HERE / "bootstrap_colab.py", "--drive-root", args.drive_root,
                 "--profile", "ref2va", "--skip-models"]
    run(bootstrap + (["--prepare-on-cpu"] if args.dry_run else []))

    total, source_w, source_h = probe(args.video)
    available = total - args.start
    seconds = min(args.duration or available, available, MAX_SECONDS)
    if seconds < 2:
        raise SystemExit(f"参考视频可用部分只有 {available:.1f} 秒，至少要 2 秒")
    orientation = args.orientation
    if orientation == "auto":
        orientation = "landscape" if source_w > source_h else "portrait"
    (width, height), (ref_w, ref_h) = SIZES[orientation]
    reference = work / "motion_small.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", args.start, "-t", seconds, "-i", args.video,
         "-vf", f"scale={ref_w}:{ref_h}:force_original_aspect_ratio=decrease,"
                f"pad={ref_w}:{ref_h}:(ow-iw)/2:(oh-ih)/2,fps=24",
         "-c:v", "libx264", "-crf", "18", "-c:a", "aac", reference])

    shutil.copy2(reference, out_dir / reference.name)
    shutil.copy2(args.prompt_file, out_dir / "prompt.txt")

    seeds = parse_seeds(args.seeds, args.draws)[:1 if args.dry_run else None]
    log = args.drive_root / "logs" / "comfyui.log"
    restart_between = host_ram_gib() < 80  # below this one 40-47 GB model staging per process
    env = os.environ.copy()
    env["H3_SOURCE_ROOT"] = str(SOURCE_ROOT)
    summary = {"job_id": args.job_id, "seconds": round(seconds, 2), "size": [width, height],
               "video_mode": args.video_mode, "dry_run": args.dry_run, "out_dir": str(out_dir),
               "seeds": seeds, "draws": []}
    for index, seed in enumerate(seeds):
        if not args.dry_run and (index == 0 or restart_between):
            restart_comfy(args.drive_root)
        offset = log.stat().st_size if log.is_file() else 0
        # One Director project per draw, so the seeds never overwrite each other.
        name = f"{args.job_id}_s{seed}"
        result = work / f"{name}.mp4"
        driver = [RUNTIME_PYTHON, HERE / "motion_transfer.py",
                  *[part for image in args.image for part in ("--image", image)],
                  "--video", reference, "--prompt-file", args.prompt_file, "--job-id", name,
                  "--duration", round(seconds, 2), "--width", width, "--height", height,
                  "--video-mode", args.video_mode, "--video-fps", 24, "--seed", seed, "--out", result]
        if args.no_video_audio:
            driver.append("--no-video-audio")
        if args.dry_run:
            api_prompt = out_dir / "api_prompt.json"
            run(driver + ["--dry-run", api_prompt], env=env)
            # ComfyUI's own validate_prompt, in a separate process that never starts the server.
            report = out_dir / "validation.json"
            run([RUNTIME_PYTHON, HERE / "validate_api_prompt.py", api_prompt, report], cwd="/content/ComfyUI")
            summary["validation"] = json.loads(report.read_text(encoding="utf-8"))["summary"]
            break
        started = time.time()
        run(driver, env=env)
        draw = {"seed": seed, "file": result.name, "elapsed_seconds": round(time.time() - started, 1),
                "log_problems": check_log(log, offset, not args.no_video_audio)}
        summary["draws"].append(draw)
        shutil.copy2(result, out_dir / result.name)
        if args.mux_original_audio:
            muxed = out_dir / f"{name}_original_audio.mp4"
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", result, "-i", reference,
                 "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", muxed])
        (out_dir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if draw["log_problems"]:
            break  # every later seed would miss the same reference; stop spending GPU time
    (out_dir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = (any(d["log_problems"] for d in summary["draws"])
              or not summary.get("validation", {}).get("all_ok", True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
