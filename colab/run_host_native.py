#!/usr/bin/env python3
"""Run the full-size, one-pass, dub-driven H3 host clip (ai_hosting's h3_host_native.py) on Colab.

That script drives the h3-dub pipeline: comfyui_quick_workflow at feat/ref2va-voice-audio,
whose only change over the runtime's own copy is in src/ (dub_audio, dub_mode).  The
pipeline is private, so it comes from a tarball in Drive (a git archive of that commit) and
is unpacked next to the runtime's copy; ComfyUI and the Director stay the runtime's own.

Steps: restore the runtime (ref2va profile, models from Drive), patch the Director as the
motion route does, unpack the pipeline and give it the runtime's Colab config, then either
--dry-run (write the graph and check it with ComfyUI's validate_prompt, fine on CPU) or start
a fresh ComfyUI and make the clip into MyDrive/minimaxH3_colab/deliveries/host-native/<id>/.
Arguments after the known ones (e.g. --steps 20) go to h3_host_native.py unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

from run_dance_swap import has_gpu, restart_comfy, run

HERE = Path(__file__).resolve().parent
COMFY = Path("/content/ComfyUI")
RUNTIME_PYTHON = Path("/content/h3env/bin/python")
RUNTIME_SOURCE = Path("/content/comfyui_quick_workflow")
PIPELINE = Path("/content/h3-dub")
COLAB_CONFIG = Path("config") / "production.colab.yaml"


def unpack(archive: Path, dest: Path) -> None:
    """A fresh copy of the pipeline; refuses members that would land outside dest."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    root = dest.resolve()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (dest / member.name).resolve()
            if target != root and root not in target.parents:
                raise SystemExit(f"流水线压缩包里有越界路径：{member.name}")
        bundle.extractall(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="主播口播：原尺寸一次成片，配音当原声，不二采不超分")
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/minimaxH3_colab"))
    parser.add_argument("--inputs", type=Path, help="portrait.png、配音、prompt.txt、h3_host_native.py 所在目录；"
                                                   "默认 <drive-root>/assets/host-native")
    parser.add_argument("--pipeline-archive", type=Path,
                        help="h3-dub 流水线的 git archive；默认 <drive-root>/pipelines/h3-dub-32c9c6a.tar.gz")
    parser.add_argument("--dub", default="host-dub-01.wav", help="inputs 目录里的配音文件名")
    parser.add_argument("--id", default="host-dub-01")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="只生成图并校验，不出片；CPU 上就能跑")
    args, extra = parser.parse_known_args()

    inputs = args.inputs or args.drive_root / "assets" / "host-native"
    archive = args.pipeline_archive or args.drive_root / "pipelines" / "h3-dub-32c9c6a.tar.gz"
    files = {"script": inputs / "h3_host_native.py", "image": inputs / "portrait.png",
             "dub": inputs / args.dub, "prompt": inputs / "prompt.txt", "archive": archive}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("找不到文件：" + "、".join(missing))
    gpu = has_gpu()
    if not args.dry_run and not gpu:
        raise SystemExit("当前没有 GPU；只校验请加 --dry-run")

    run([sys.executable, HERE / "bootstrap_colab.py", "--drive-root", args.drive_root,
         "--profile", "ref2va", "--skip-models", *([] if gpu else ["--prepare-on-cpu"])])
    run([sys.executable, HERE / "patch_director.py"])
    unpack(archive, PIPELINE)
    # The runtime's Colab config (Drive paths, RAM-safe upscale batches) fits the dub branch too:
    # that branch added job fields, not config keys.
    shutil.copy2(RUNTIME_SOURCE / COLAB_CONFIG, PIPELINE / COLAB_CONFIG)

    out_dir = (Path("/content/h3-host-native") / args.id / "check" if args.dry_run
               else args.drive_root / "deliveries" / "host-native" / args.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [RUNTIME_PYTHON, files["script"], "--pipeline", PIPELINE, "--config", COLAB_CONFIG,
           "--image", files["image"], "--dub", files["dub"], "--prompt-file", files["prompt"],
           "--out-dir", out_dir, "--id", args.id, *(["--seed", args.seed] if args.seed is not None else []),
           *extra]

    if args.dry_run:
        run([*cmd, "--dry-run"])
        api, report = out_dir / f"{args.id}.api.json", out_dir / "validation.json"
        run([RUNTIME_PYTHON, HERE / "validate_api_prompt.py", api, report], cwd=COMFY)
        result = json.loads(report.read_text(encoding="utf-8"))
        if not result["all_ok"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({"id": args.id, "dry_run": True, "validation_ok": result["all_ok"]}, ensure_ascii=False))
        return 0 if result["all_ok"] else 1

    restart_comfy(args.drive_root)
    run(cmd)
    clip = out_dir / f"{args.id}.mp4"
    # Contact sheet in Drive, so the clip can be looked at after the GPU is released.
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vf", "fps=1,scale=270:-2,tile=8x2",
         "-frames:v", "1", out_dir / "peek.jpg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
