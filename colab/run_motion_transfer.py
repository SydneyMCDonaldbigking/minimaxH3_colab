#!/usr/bin/env python3
"""Run one H3 motion-transfer job end to end on a Colab runtime.

Appearance comes from --image (the character, and the scene it stands in), motion
from --video (the dance). The whole frame is regenerated, unlike the dance_swap
canvas, which only repaints one person inside the original video. --camera-video
adds a second reference in camera mode, so the shot can move (a walking selfie,
say) while the dance still comes from a locked-off clip.

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
import itertools
import json
import math
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
# Ref2VA caps a segment and its reference video at 15 s, but a 15 s segment renders 362 frames
# while the Director cuts a reference to 360 and then down to 345: 14 s keeps the two equal.
MAX_SEGMENT_SECONDS = 14
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
    """Duration in seconds and display width/height of the first video stream (its own duration if known)."""
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,duration:stream_side_data=rotation:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout)
    stream = info["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = abs(int(next((d.get("rotation", 0) for d in stream.get("side_data_list", [])), 0)))
    if rotation in (90, 270):  # phone clips are often stored landscape with a rotation flag
        width, height = height, width
    return float(stream.get("duration") or info["format"]["duration"]), width, height


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode == 0


def check_prompt(text: str, *modes: str) -> None:
    if "retention_analysis" not in text or "<Video" not in text:
        return
    for mode in modes:
        if MODE_PHRASES[mode] not in text:
            raise SystemExit(
                f"提示词写了 retention_analysis，但没有 {mode} 模式的官方声明原句"
                f"（需包含 “{MODE_PHRASES[mode]}”）；不补这句这一路参考视频的模式不会进模型。")


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


def check_log(log: Path, offset: int, want_audio: bool, want_camera: bool = False) -> list[str]:
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
    motion = [line for line in videos if "<Video 1>" in line]
    if not motion:
        problems.append("没有“参考视频 <Video 1>”：video_refs 没生效")
    elif want_audio and not all("已发送原音轨" in line for line in motion):
        problems.append("参考视频只送了画面，音轨没发送")
    if want_camera and not any("<Video 2>" in line for line in videos):
        problems.append("没有“参考视频 <Video 2>”：运镜参考没生效")
    return problems


def director_frames(seconds: int) -> int:
    """Frames the Director renders for a segment of this many whole seconds.

    Same rule as studio_node._segment_frame_count: the nearest 17k + 5 count from 39 to 362.
    The Director first rounds a segment to whole seconds, so 9.42 s renders as 9 s, 209 frames.
    Cutting each reference part to exactly this many frames keeps the segments on the source
    timeline frame for frame, so the source audio can be laid back over the result.
    """
    return min(range(39, 363, 17), key=lambda frames: abs(frames - seconds * 24))


def plan_segments(available: float, duration: float | None, segments: int) -> list[int]:
    """Whole seconds per segment, covering as much of the clip as fits without running past it."""
    limit = int(min(duration or available, available) * 24 + 0.05)  # 18.833333 s is 452 frames, not 451
    longest = director_frames(MAX_SEGMENT_SECONDS)
    count = segments or max(1, math.ceil(limit / longest))
    if count * longest < limit:
        raise SystemExit(f"{count} 段最多 {count * longest / 24:.1f} 秒，参考视频要用 {limit / 24:.1f} 秒；"
                         "请加大 --segments 或用 --duration 截短")
    plans = [plan for plan in itertools.combinations_with_replacement(range(2, MAX_SEGMENT_SECONDS + 1), count)
             if sum(map(director_frames, plan)) <= limit]
    if not plans:
        raise SystemExit(f"参考视频只有 {limit / 24:.2f} 秒，不够分 {count} 段；请减少 --segments 或换更长的参考视频")
    # Most frames covered first, then the most even split.
    return list(max(plans, key=lambda plan: (sum(map(director_frames, plan)), plan[0] - plan[-1])))


def cut(source: Path, start: float, frames: int, out: Path, size: tuple[int, int], audio: bool) -> Path:
    """Exactly `frames` frames at 24 fps from `start`, letterboxed to `size`."""
    width, height = size
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.4f}", "-t", f"{(frames + 0.5) / 24:.4f}",
         "-i", source, "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps=24",
         "-frames:v", frames, "-c:v", "libx264", "-crf", "18", *(["-c:a", "aac"] if audio else ["-an"]), out])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="H3 动作迁移：角色图定外观和场景，舞蹈视频定动作")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/minimaxH3_colab"))
    parser.add_argument("--image", type=Path, action="append", required=True,
                        help="参考图，可重复；顺序就是 <Picture 1>、<Picture 2>…")
    parser.add_argument("--video", type=Path, required=True, help="舞蹈参考视频（单个连续镜头）")
    parser.add_argument("--camera-video", type=Path,
                        help="运镜参考视频，作 <Video 2>、camera 模式、不送音轨：只照它的机位走，不取它的人和景")
    parser.add_argument("--camera-start", type=float,
                        help="运镜视频第几秒对上舞蹈视频的 --start，默认同 --start")
    parser.add_argument("--prompt-file", type=Path, action="append", required=True,
                        help="提示词；给一个各段共用，给的个数等于段数时每段一个。"
                             "都按段 1 的图片编号写，段 2 起脚本自动顺延")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="从参考视频第几秒开始取")
    parser.add_argument("--duration", type=float, help="成片秒数，默认取参考视频全长")
    parser.add_argument("--segments", type=int, default=0,
                        help="分几段生成，默认按每段不超过 15 秒自动算；段 2 起接上一段尾帧")
    parser.add_argument("--orientation", choices=["auto", *sorted(SIZES)], default="auto",
                        help="成片横竖屏，默认跟参考视频一致")
    parser.add_argument("--video-mode", choices=sorted(MODE_PHRASES), default="action")
    parser.add_argument("--no-video-audio", action="store_true", help="不把参考视频的音轨送进模型")
    parser.add_argument("--mux-original-audio", action="store_true",
                        help="另存一份用参考视频原音轨整条覆盖的成片（分段时总是另存）")
    parser.add_argument("--seeds", help="逗号分隔的 seed，例如 101,102,103；每个 seed 出一条")
    parser.add_argument("--draws", type=int, default=1, help="不给 --seeds 时随机抽几条")
    parser.add_argument("--dry-run", action="store_true", help="CPU 检查：只生成 API prompt，不生成视频")
    args = parser.parse_args()

    modes = [args.video_mode, *(["camera"] if args.camera_video else [])]
    for path in [*args.image, args.video, *args.prompt_file, *([args.camera_video] if args.camera_video else [])]:
        if not path.is_file():
            raise SystemExit(f"找不到文件：{path}")
    for prompt_file in args.prompt_file:
        check_prompt(prompt_file.read_text(encoding="utf-8"), *modes)
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
    # Before ComfyUI starts: the snapshot's Director calls ReferenceToVideo in the pre-bc5f880 order.
    run([sys.executable, HERE / "patch_director.py"])

    total, source_w, source_h = probe(args.video)
    plan = plan_segments(total - args.start, args.duration, args.segments)
    frames = [director_frames(seconds) for seconds in plan]
    count, seconds = len(plan), sum(frames) / 24
    if len(args.prompt_file) not in (1, count):
        raise SystemExit(f"给了 {len(args.prompt_file)} 个提示词，但要分 {count} 段；给一个共用，或者每段一个")
    orientation = args.orientation
    if orientation == "auto":
        orientation = "landscape" if source_w > source_h else "portrait"
    (width, height), ref_size = SIZES[orientation]
    reference = cut(args.video, args.start, sum(frames), work / "motion_small.mp4", ref_size, audio=True)
    camera = None
    if args.camera_video:
        camera_start = args.start if args.camera_start is None else args.camera_start
        camera_left = probe(args.camera_video)[0] - camera_start
        if camera_left * 24 < sum(frames) - 1:
            raise SystemExit(f"运镜视频从第 {camera_start} 秒起只剩 {camera_left:.2f} 秒，不够 {seconds:.2f} 秒")
        camera = cut(args.camera_video, camera_start, sum(frames), work / "camera_small.mp4", ref_size, audio=False)
    # Segment N gets frames [sum(frames[:N-1]), sum(frames[:N])) of every reference.
    starts = [sum(frames[:index]) / 24 for index in range(count)]
    parts = [cut(reference, start, n, work / f"motion_part{index + 1}.mp4", ref_size, audio=True)
             for index, (start, n) in enumerate(zip(starts, frames))] if count > 1 else []
    camera_parts = [cut(camera, start, n, work / f"camera_part{index + 1}.mp4", ref_size, audio=False)
                    for index, (start, n) in enumerate(zip(starts, frames))] if camera and count > 1 else []

    for source in (reference, camera):
        if source:
            shutil.copy2(source, out_dir / source.name)
    for index, prompt_file in enumerate(args.prompt_file):
        name = "prompt.txt" if len(args.prompt_file) == 1 else f"prompt_seg{index + 1}.txt"
        shutil.copy2(prompt_file, out_dir / name)

    seeds = parse_seeds(args.seeds, args.draws)[:1 if args.dry_run else None]
    log = args.drive_root / "logs" / "comfyui.log"
    restart_between = host_ram_gib() < 80  # below this one 40-47 GB model staging per process
    env = os.environ.copy()
    env["H3_SOURCE_ROOT"] = str(SOURCE_ROOT)
    summary = {"job_id": args.job_id, "seconds": round(seconds, 3), "segments": count,
               "segment_seconds": plan, "segment_frames": frames, "size": [width, height],
               "video_mode": args.video_mode, "camera_video": args.camera_video.name if camera else None,
               "dry_run": args.dry_run, "out_dir": str(out_dir), "seeds": seeds, "draws": []}
    for index, seed in enumerate(seeds):
        if not args.dry_run and (index == 0 or restart_between):
            restart_comfy(args.drive_root)
        offset = log.stat().st_size if log.is_file() else 0
        # One Director project per draw, so the seeds never overwrite each other.
        name = f"{args.job_id}_s{seed}"
        result = work / f"{name}.mp4"
        driver = [RUNTIME_PYTHON, HERE / "motion_transfer.py",
                  *[part for image in args.image for part in ("--image", image)],
                  "--prompt-file", args.prompt_file[0], "--job-id", name, "--width", width, "--height", height,
                  "--video-mode", args.video_mode, "--video-fps", 24, "--seed", seed, "--out", result]
        if len(args.prompt_file) > 1:
            driver += [arg for prompt_file in args.prompt_file for arg in ("--segment-prompt", prompt_file)]
        if parts:
            # Whole seconds: the Director rounds them anyway, and each part is cut to what it renders.
            driver += [arg for part, secs in zip(parts, plan) for arg in ("--slice", f"{part}:{secs}")]
            driver += [arg for part in camera_parts for arg in ("--camera-video", part)]
        else:
            driver += ["--video", reference, "--duration", plan[0]]
            driver += ["--camera-video", camera] if camera else []
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
        if parts and restart_between:
            # Below 80 GB a second model staging in one process gets OOM-killed: one segment per
            # process, chained through the tail frame the Director left on disk, then merge.
            for segment in range(1, count + 1):
                if segment > 1:
                    restart_comfy(args.drive_root)
                run(driver + ["--only", segment], env=env)
            run(driver + ["--merge-only"], env=env)
        else:
            run(driver, env=env)
        draw = {"seed": seed, "file": result.name, "elapsed_seconds": round(time.time() - started, 1),
                "log_problems": check_log(log, offset, not args.no_video_audio, want_camera=bool(camera))}
        summary["draws"].append(draw)
        shutil.copy2(result, out_dir / result.name)
        if args.mux_original_audio or parts:
            # Segments get their music separately and it jumps at the seam; lay the source track over.
            # The parts were cut to the frames the Director renders, so the track lines up throughout.
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
