#!/usr/bin/env python3
"""Run the FastH3 OmniRef character-swap canvas headless on a Colab runtime.

The canvas keeps the source video and repaints one dancer: SAM 3.1 tracks the
person named by --target, Character Swap V1.5 on FastH3 OmniRef redraws them as
the character in --image, and the result is pasted back into the original frame.

Steps: restore the runtime from the Drive snapshot (dance_swap profile), copy
the inputs into ComfyUI's input folder, fill them into the canvas by node type,
turn the canvas into an API prompt and validate it with ComfyUI's own
validate_prompt (colab/canvas_api.py), then queue one run per seed and save each
clip under MyDrive/minimaxH3_colab/deliveries/dance_swap/<job-id>/.

Drive must already be mounted at /content/drive.  --dry-run runs on CPU and stops
after validation.
"""
from __future__ import annotations

import argparse
import json
import math
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMFY = Path("/content/ComfyUI")
RUNTIME_PYTHON = Path("/content/h3env/bin/python")
CANVAS = COMFY / "user" / "default" / "workflows" / "H3-Character-Swap-1.5-SAM3.1-FastH3-OmniRef.json"
BASE_URL = "http://127.0.0.1:8188"
MAX_SECONDS = 15.0
JOB_TIMEOUT_SECONDS = 3 * 3600
# ResolutionSelector's aspect options; canvas_api.py expands "4:3" to "4:3 (Standard)".
ASPECTS = {"1:1": 1.0, "2:3": 2 / 3, "3:2": 3 / 2, "3:4": 3 / 4, "4:3": 4 / 3,
           "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9}


def aligned_frames(seconds: float) -> int:
    """Largest H3 frame count (17k + 5 at 24 fps) that fits in this many seconds."""
    return max(5, (int(seconds * 24) - 5) // 17 * 17 + 5)


def nearest_aspect(width: int, height: int) -> str:
    """The canvas aspect closest to the source, so the centre crop removes as little as possible."""
    return min(ASPECTS, key=lambda name: abs(math.log(width / height / ASPECTS[name])))


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


def only(nodes: list[dict], kind: str, active: bool = True) -> dict:
    """The single node of this type (ignoring bypassed and muted nodes when active)."""
    matches = [n for n in nodes if n["type"] == kind and (not active or n.get("mode", 0) == 0)]
    if len(matches) != 1:
        raise SystemExit(f"画布里应有且只有一个启用的 {kind}，找到 {len(matches)} 个；这不是预期的换人画布")
    return matches[0]


def fill_canvas(canvas: dict, *, video: str, image: str, target: str, prompt: str,
                seconds: float, start_frames: int, aspect: str, sam_max_objects: int = 4) -> dict:
    """Write the job's inputs into the canvas, finding each node by its type."""
    nodes = canvas["nodes"]
    loader = only(nodes, "VHS_LoadVideo")
    loader["widgets_values"].update({"video": video, "skip_first_frames": start_frames})
    only(nodes, "LoadImage")["widgets_values"][0] = image
    only(nodes, "CLIPTextEncode")["widgets_values"][0] = target
    # [detection_threshold, max_objects, detect_interval]. With one object, a dancer who covers
    # their face has the head split off as a second track that never makes it into the mask,
    # so the original head stays in the frame. The masks of all tracks are merged.
    only(nodes, "SAM3_VideoTrack")["widgets_values"][1] = sam_max_objects
    only(nodes, "PrimitiveStringMultiline")["widgets_values"][0] = prompt
    only(nodes, "PrimitiveFloat")["widgets_values"][0] = round(seconds, 2)
    # The one PrimitiveNode drives both ResolutionSelector aspect_ratio inputs.
    only(nodes, "PrimitiveNode")["widgets_values"][0] = aspect
    return canvas


def api_node(prompt: dict, kind: str) -> str:
    ids = [key for key, item in prompt.items() if item["class_type"] == kind]
    if len(ids) != 1:
        raise SystemExit(f"API prompt 里应有且只有一个 {kind}，找到 {len(ids)} 个")
    return ids[0]


def comfy_json(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE_URL + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def run_prompt(prompt: dict) -> dict:
    """Queue the prompt and wait for its history entry."""
    try:
        queued = comfy_json("/prompt", {"prompt": prompt})
    except urllib.error.HTTPError as error:
        raise SystemExit(f"ComfyUI 拒绝任务：{error.read().decode('utf-8', 'replace')[:2000]}") from error
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        entry = comfy_json(f"/history/{prompt_id}").get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise SystemExit(f"ComfyUI 执行失败：{json.dumps(status, ensure_ascii=False)[:2000]}")
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(5)
    raise SystemExit(f"ComfyUI 任务超时：{prompt_id}")


def download_video(entry: dict, node_id: str, destination: Path) -> None:
    files = entry["outputs"].get(node_id, {}).get("gifs") or []
    if not files:
        raise SystemExit("任务完成了，但 VHS_VideoCombine 没有输出视频")
    item = files[-1]
    query = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
    with urllib.request.urlopen(f"{BASE_URL}/view?{query}", timeout=600) as response, \
            destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="H3 换人：保留原视频，只把一个舞者换成角色图里的人")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/minimaxH3_colab"))
    parser.add_argument("--canvas", type=Path, default=CANVAS, help="FastH3 OmniRef 换人画布 JSON")
    parser.add_argument("--video", type=Path, required=True, help="原舞蹈视频（单个连续镜头）")
    parser.add_argument("--image", type=Path, required=True, help="新角色图")
    parser.add_argument("--target", required=True, help="给 SAM 的英文描述：原视频里要换掉的是哪个人")
    parser.add_argument("--prompt-file", type=Path, required=True, help="H3 提示词（新角色的 subject_definitions 等）")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="从原视频第几秒开始")
    parser.add_argument("--duration", type=float, help="成片秒数，默认取原视频长度，最多 15")
    parser.add_argument("--aspect", choices=["auto", *ASPECTS], default="auto",
                        help="成片比例，默认取最接近原视频的")
    parser.add_argument("--sam-max-objects", type=int, default=4,
                        help="SAM 最多跟几个目标，遮罩取全部目标的并集；原画布是 1，"
                             "舞者挡脸时头会被拆成另一个目标而漏掉。画面里有别人入镜时可调回 1")
    parser.add_argument("--seeds", help="逗号分隔的 seed，例如 101,102,103；每个 seed 出一条")
    parser.add_argument("--draws", type=int, default=1, help="不给 --seeds 时随机抽几条")
    parser.add_argument("--dry-run", action="store_true", help="CPU 检查：只转换并校验，不生成视频")
    args = parser.parse_args()

    for path in (args.video, args.image, args.prompt_file):
        if not path.is_file():
            raise SystemExit(f"找不到文件：{path}")
    if not args.dry_run and not has_gpu():
        raise SystemExit("当前没有 GPU；只做检查请加 --dry-run")

    # --skip-models: the dance_swap files are already in Drive; re-hashing the 22 GB FastH3
    # file every session would only spend GPU minutes. ComfyUI rejects a missing file anyway.
    run([sys.executable, HERE / "bootstrap_colab.py", "--drive-root", args.drive_root,
         "--profile", "dance_swap", "--skip-models", *(["--prepare-on-cpu"] if args.dry_run else [])])
    if not args.canvas.is_file():
        raise SystemExit(f"找不到换人画布：{args.canvas}")

    total, width, height = probe(args.video)
    available = total - args.start
    # The canvas rounds its frame count up to the H3 grid (17k + 5 at 24 fps); asking for more
    # frames than the source holds would leave the reference short, so snap down first.
    seconds = aligned_frames(min(args.duration or available, available, MAX_SECONDS)) / 24
    if seconds < 2:
        raise SystemExit(f"原视频可用部分只有 {available:.1f} 秒，至少要 2 秒")
    aspect = args.aspect if args.aspect != "auto" else nearest_aspect(width, height)

    work = Path("/content/h3-dance-swap") / args.job_id
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "check" if args.dry_run else args.drive_root / "deliveries" / "dance_swap" / args.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = COMFY / "input"
    video_name = f"{args.job_id}_source{args.video.suffix.lower()}"
    image_name = f"{args.job_id}_character{args.image.suffix.lower()}"
    shutil.copy2(args.video, inputs / video_name)
    shutil.copy2(args.image, inputs / image_name)

    canvas = fill_canvas(
        json.loads(args.canvas.read_text(encoding="utf-8-sig")),
        video=video_name, image=image_name, target=args.target.strip(),
        prompt=args.prompt_file.read_text(encoding="utf-8"), seconds=seconds,
        start_frames=round(args.start * 24), aspect=aspect, sam_max_objects=args.sam_max_objects,
    )
    filled = work / "canvas.json"
    filled.write_text(json.dumps(canvas, ensure_ascii=False, indent=2), encoding="utf-8")
    api_path, report_path = work / "api_prompt.json", work / "validation.json"
    run([RUNTIME_PYTHON, HERE / "canvas_api.py", filled, api_path, report_path, inputs], cwd=COMFY)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name in ("canvas.json", "api_prompt.json", "validation.json"):
        shutil.copy2(work / name, out_dir / name)
    shutil.copy2(args.prompt_file, out_dir / "prompt.txt")

    seeds = parse_seeds(args.seeds, args.draws)
    summary = {"job_id": args.job_id, "seconds": round(seconds, 2), "aspect": aspect,
               "target": args.target, "dry_run": args.dry_run, "out_dir": str(out_dir),
               "validation_ok": report["all_ok"], "seeds": [] if args.dry_run else seeds, "draws": []}
    if not report["all_ok"]:
        print(json.dumps(report["validate_prompt"]["node_errors"], ensure_ascii=False, indent=2))
    if args.dry_run or not report["all_ok"]:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if report["all_ok"] else 1

    prompt = json.loads(api_path.read_text(encoding="utf-8"))
    noise_id, combine_id = api_node(prompt, "RandomNoise"), api_node(prompt, "VHS_VideoCombine")
    restart_between = host_ram_gib() < 80  # below this one model staging per process
    for index, seed in enumerate(seeds):
        if index == 0 or restart_between:
            restart_comfy(args.drive_root)
        name = f"{args.job_id}_s{seed}"
        prompt[noise_id]["inputs"]["noise_seed"] = seed
        prompt[combine_id]["inputs"]["filename_prefix"] = f"dance_swap/{name}"
        started = time.time()
        entry = run_prompt(prompt)
        result = out_dir / f"{name}.mp4"
        download_video(entry, combine_id, result)
        summary["draws"].append({"seed": seed, "file": result.name,
                                 "elapsed_seconds": round(time.time() - started, 1)})
        (out_dir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
