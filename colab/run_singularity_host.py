#!/usr/bin/env python3
"""Make a dub-driven host clip with the "Singularity V2 face refine" workflow on a Colab GPU.

The canvas (built from the RunningHub workflow, kept in Drive with the portrait, the dub and
the prompt) generates with the community Singularity v2 Ref2VA weights, the turbo LoRA and
the SelfLift progressive sampler at about 1.5 MP, then tracks the face, restores the face
crops with VOSR 2.0 and stitches them back.  The dub is <Audio 1>, and the original dub is
laid back over the result, as production does.

Its three custom node packs are installed at pinned commits on every run (the runtime
snapshot does not carry them).  The Singularity weights (21 GB) and VOSR 2.0 (7 GB) do not
fit in Drive, so they are downloaded from Hugging Face to the runtime's own disk; ComfyUI
finds the weights through extra_model_paths.yaml.

Steps: restore the runtime (ref2va profile), install the node packs, cut the dub window,
fill the canvas, convert it and check it with validate_prompt (with a placeholder for the
weights when they are not downloaded yet), then download, start a fresh ComfyUI, run, and
save both clips to MyDrive/minimaxH3_colab/deliveries/host-singularity/<id>/.
--dry-run stops after the check and downloads nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from run_dance_swap import BASE_URL, has_gpu, only, probe, restart_comfy, run, run_prompt

HERE = Path(__file__).resolve().parent
COMFY = Path("/content/ComfyUI")
RUNTIME_PYTHON = Path("/content/h3env/bin/python")
LOCAL_MODELS = Path("/content/local_models")
SINGULARITY = ("WarmBloodAban/Minimax-h3_Singularity", "Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors")
FACE_DETECTOR = ("Bingsu/adetailer", "face_yolov8n.pt")
NODE_PACKS = {  # reviewed 2026-10-27: no subprocess/exec; VOSR2 only downloads its own weights
    "ComfyUI-H3-FaceRefine": ("https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git",
                              "d8521d14fe0d721d80cd9417fff5a559cbc21aba"),
    "ComfyUI-VOSR2": ("https://github.com/ylchen333/ComfyUI-VOSR2.git", "6a21810b3f4f1a0ffc477c55330db0ac03c671f0"),
    "SelfLiftH3Sampler": ("https://github.com/rogen84/SelfLiftH3Sampler.git",
                          "cb772ed2a02266c9b6efefc149dae2d731d78ded"),
}
# FaceRefine wants ultralytics (and scipy); insightface is only for identity tracking, which
# the canvas leaves off.  These must not move the runtime's torch stack.
PINNED = ("torch", "torchvision", "torchaudio", "numpy", "pillow")
# FaceRefine lists face detectors only from an "ultralytics_bbox" (or "ultralytics") model folder,
# which the Impact subpack usually registers; this runtime has no Impact, so register it here.
EXTRA_PATHS = f"""
colab_local_models:
    base_path: {LOCAL_MODELS}
    diffusion_models: diffusion_models
    ultralytics_bbox: ultralytics/bbox
"""


def install_node_packs() -> None:
    for name, (url, commit) in NODE_PACKS.items():
        dest = COMFY / "custom_nodes" / name
        if not (dest / ".git").is_dir():
            run(["git", "clone", "-q", "--filter=blob:none", url, dest])
        run(["git", "-C", dest, "checkout", "-q", commit])
    freeze = subprocess.run([str(RUNTIME_PYTHON), "-m", "pip", "freeze"], capture_output=True, text=True,
                            check=True).stdout
    constraints = Path("/content/h3-singularity-constraints.txt")
    constraints.write_text("\n".join(line for line in freeze.splitlines()
                                     if line.split("==")[0].lower() in PINNED) + "\n", encoding="utf-8")
    run([RUNTIME_PYTHON, "-m", "pip", "install", "-q", "ultralytics", "scipy", "-c", constraints])
    config = COMFY / "extra_model_paths.yaml"
    text = config.read_text(encoding="utf-8") if config.is_file() else ""
    if "colab_local_models:" not in text:
        config.write_text(text + EXTRA_PATHS, encoding="utf-8")


def hf_download(repo: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.unlink(missing_ok=True)  # an empty placeholder from a dry run
    code = ("import sys; from huggingface_hub import hf_hub_download; "
            "print(hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))")
    run([RUNTIME_PYTHON, "-c", code, repo, filename, dest_dir])
    return target


def audio_seconds(media: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(media)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def window_seconds(duration: float, start: float, length: float | None) -> float:
    """Seconds of dub the clip covers, and so the generation length (H3 rounds up itself)."""
    return max(0.5, (duration - start) if length is None else min(length, duration - start))


def fill_canvas(canvas: dict, *, image: str, audio: str, prompt: str, seconds: float, seed: int,
                megapixels: float, prefix: str) -> dict:
    nodes = canvas["nodes"]
    only(nodes, "LoadImage")["widgets_values"][0] = image
    only(nodes, "LoadAudio")["widgets_values"][0] = audio
    only(nodes, "PrimitiveStringMultiline")["widgets_values"][0] = prompt
    only(nodes, "PrimitiveFloat")["widgets_values"][0] = float(math.ceil(seconds))
    only(nodes, "SelfLiftH3Sampler")["widgets_values"][0] = seed
    only(nodes, "ResolutionSelector")["widgets_values"][1] = megapixels
    for node in nodes:
        if node["type"] == "VHS_VideoCombine":
            stage = "face" if node["id"] == 270 else "raw"
            node["widgets_values"]["filename_prefix"] = f"{prefix}_{stage}"
    return canvas


def download_output(entry: dict, node_id: str, destination: Path) -> Path:
    files = entry["outputs"].get(node_id, {}).get("gifs") or []
    if not files:
        raise SystemExit(f"节点 {node_id} 没有输出视频")
    item = files[-1]
    query = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
    with urllib.request.urlopen(f"{BASE_URL}/view?{query}", timeout=600) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="主播口播：Singularity V2 + SelfLift + 脸部 VOSR2 修复")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/minimaxH3_colab"))
    parser.add_argument("--inputs", type=Path, help="canvas.json、prompt.txt、portrait.png、配音所在目录；"
                                                   "默认 <drive-root>/assets/host-singularity")
    parser.add_argument("--dub", default="host-dub-01.wav")
    parser.add_argument("--id", default="host-dub-01")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--megapixels", type=float, default=1.5, help="生成尺寸（百万像素），9:16；1.5 ≈ 928x1664")
    parser.add_argument("--audio-start", type=float, default=0.0, help="从配音第几秒开始（分段出片用）")
    parser.add_argument("--audio-length", type=float, help="这一段用几秒配音；默认到结尾")
    parser.add_argument("--dry-run", action="store_true", help="只装节点、填图、校验，不下载模型、不出片")
    args = parser.parse_args()

    inputs = args.inputs or args.drive_root / "assets" / "host-singularity"
    files = {name: inputs / name for name in ("canvas.json", "prompt.txt", "portrait.png", args.dub)}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("找不到文件：" + "、".join(missing))
    gpu = has_gpu()
    if not args.dry_run and not gpu:
        raise SystemExit("当前没有 GPU；只校验请加 --dry-run")

    run([sys.executable, HERE / "bootstrap_colab.py", "--drive-root", args.drive_root,
         "--profile", "ref2va", "--skip-models", *([] if gpu else ["--prepare-on-cpu"])])
    install_node_packs()

    work = Path("/content/h3-singularity") / args.id
    work.mkdir(parents=True, exist_ok=True)
    comfy_inputs = COMFY / "input"
    image = f"{args.id}_portrait.png"
    shutil.copy2(files["portrait.png"], comfy_inputs / image)
    seconds = window_seconds(audio_seconds(files[args.dub]), args.audio_start, args.audio_length)
    dub = work / "dub.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", args.audio_start, "-t", round(seconds, 3),
         "-i", files[args.dub], "-c:a", "pcm_s16le", dub])
    audio = f"{args.id}_dub.wav"
    shutil.copy2(dub, comfy_inputs / audio)

    hf_download(*FACE_DETECTOR, LOCAL_MODELS / "ultralytics" / "bbox")  # 6 MB; its name must be listed to validate
    weights = LOCAL_MODELS / "diffusion_models" / SINGULARITY[1]
    if not weights.is_file():  # validate_prompt only checks the name is listed
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.touch()
    canvas = fill_canvas(json.loads(files["canvas.json"].read_text(encoding="utf-8")), image=image, audio=audio,
                         prompt=files["prompt.txt"].read_text(encoding="utf-8"), seconds=seconds, seed=args.seed,
                         megapixels=args.megapixels, prefix=f"host-singularity/{args.id}")
    filled, api_path, report_path = work / "canvas.json", work / "api_prompt.json", work / "validation.json"
    filled.write_text(json.dumps(canvas, ensure_ascii=False, indent=1), encoding="utf-8")
    run([RUNTIME_PYTHON, HERE / "canvas_api.py", filled, api_path, report_path, comfy_inputs], cwd=COMFY)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {"id": args.id, "seed": args.seed, "megapixels": args.megapixels, "audio_start": args.audio_start,
               "seconds": round(seconds, 3), "validation_ok": report["all_ok"], "dry_run": args.dry_run}
    if not report["all_ok"]:
        print(json.dumps(report, ensure_ascii=False, indent=2)[:8000])
    if args.dry_run or not report["all_ok"]:
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if report["all_ok"] else 1

    started = time.monotonic()
    hf_download(*SINGULARITY, weights.parent)
    summary["download_s"] = round(time.monotonic() - started, 1)  # VOSR2 fetches its own on first use

    restart_comfy(args.drive_root)
    out_dir = args.drive_root / "deliveries" / "host-singularity" / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    vram_log = out_dir / f"{args.id}.vram.csv"
    with vram_log.open("w") as log:
        watcher = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                    "-l", "1"], stdout=log)
        started = time.monotonic()
        try:
            entry = run_prompt(json.loads(api_path.read_text(encoding="utf-8")))
        finally:
            watcher.terminate()
            watcher.wait()
    summary["generate_s"] = round(time.monotonic() - started, 1)
    used = [int(line) for line in vram_log.read_text().split() if line.isdigit()]
    summary["vram_peak_gb"] = round(max(used, default=0) / 1024, 1)

    for stage, node_id in (("raw", "243"), ("face", "270")):
        made = download_output(entry, node_id, work / f"{stage}.mp4")
        # The dub laid back over the whole clip, as production does.
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", made, "-i", dub, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_dir / f"{args.id}_{stage}.mp4"])
    _, width, height = probe(out_dir / f"{args.id}_face.mp4")
    summary["size"] = f"{width}x{height}"
    shutil.copy2(files["prompt.txt"], out_dir / "prompt.txt")
    shutil.copy2(api_path, out_dir / "api_prompt.json")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_dir / f"{args.id}_face.mp4", "-vf",
         "fps=1,scale=270:-2,tile=8x2", "-frames:v", "1", out_dir / "peek.jpg"])
    (out_dir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
