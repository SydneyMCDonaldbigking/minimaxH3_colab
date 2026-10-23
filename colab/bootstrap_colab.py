#!/usr/bin/env python3
"""Install the private MiniMax H3 workflow into a Colab runtime.

The public launcher never vendors private nodes.  A small archive of the
private source and all large model/cache files live in the user's Drive.
ComfyUI and Python packages are recreated on Colab's local SSD each session.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


SOURCE_REPO = "SydneyMCDonaldbigking/comfyui_quick_workflow"
SOURCE_REF = "main"
COMFYUI_REF = "v0.34.0"
TORCH_STACK = ("torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1")
FRONTEND_STACK = (
    "comfyui-frontend-package==1.51.9",
    "comfyui-workflow-templates==0.11.54",
    "comfyui-embedded-docs==0.5.10",
    "comfy-kitchen==0.2.31",
    "comfy-aimdo==0.4.15",
)
MODEL_FOLDERS = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "latent_upscale_models",
    "upscale_models",
)


@dataclass(frozen=True)
class Layout:
    drive: Path
    content: Path

    @property
    def cache(self) -> Path:
        return self.drive / "runtime-cache"

    @property
    def models(self) -> Path:
        return self.drive / "models"

    @property
    def source(self) -> Path:
        return self.content / "comfyui_quick_workflow"

    @property
    def comfy(self) -> Path:
        return self.content / "ComfyUI"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def safe_remove(path: Path, root: Path) -> None:
    resolved = path.resolve(strict=False)
    allowed = root.resolve(strict=False)
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"拒绝删除运行目录之外的路径: {resolved}")
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def ensure_layout(layout: Layout) -> None:
    for folder in (
        layout.cache,
        layout.cache / "pip-cache",
        layout.cache / "modelscope-staging",
        layout.models,
        layout.drive / "jobs" / "inbox",
        layout.drive / "jobs" / "running",
        layout.drive / "jobs" / "done",
        layout.drive / "jobs" / "failed",
        layout.drive / "work",
        layout.drive / "assets",
        layout.drive / "review_ready",
        layout.drive / "rejected",
        layout.drive / "deliveries",
        layout.drive / "logs",
        layout.drive / "private-nodes",
    ):
        folder.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FOLDERS:
        (layout.models / name).mkdir(parents=True, exist_ok=True)


def gpu_report() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(command, text=True).strip().splitlines()[0]
        name, memory_mib, driver = [part.strip() for part in raw.split(",", 2)]
        return {"name": name, "vram_gib": round(float(memory_mib) / 1024, 2), "driver": driver}
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return {"name": None, "vram_gib": 0, "driver": None}


def host_ram_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 2)
    except (OSError, ValueError, IndexError):
        pass
    return 0


def download_source(layout: Layout, token: str | None) -> Path:
    archive = layout.cache / "private-source-main.zip"
    if token:
        url = f"https://api.github.com/repos/{SOURCE_REPO}/zipball/{SOURCE_REF}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "minimax-h3-colab-launcher/1.0",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        partial = archive.with_suffix(".zip.partial")
        try:
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            partial.replace(archive)
            print(f"私有源包已刷新到 Drive: {archive}")
        except (urllib.error.URLError, TimeoutError) as error:
            partial.unlink(missing_ok=True)
            if not archive.is_file():
                raise RuntimeError(f"无法下载私有源仓库，且 Drive 没有缓存: {error}") from error
            print(f"警告：GitHub 下载失败，改用 Drive 中的缓存源包: {error}")
    elif not archive.is_file():
        raise RuntimeError(
            "首次安装需要 Colab Secret GITHUB_TOKEN（只需私有仓库读取权限）；"
            "之后可直接使用 Drive 中的 private-source-main.zip。"
        )

    unpack = layout.content / "h3-source-unpack"
    safe_remove(unpack, layout.content)
    safe_remove(layout.source, layout.content)
    unpack.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(unpack)
    roots = [item for item in unpack.iterdir() if item.is_dir()]
    if len(roots) != 1 or not (roots[0] / "src" / "worker.py").is_file():
        raise RuntimeError("私有源包结构不正确，无法找到 src/worker.py")
    roots[0].replace(layout.source)
    safe_remove(unpack, layout.content)
    return layout.source


def install_python_stack(layout: Layout, source: Path, skip_sage: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(layout.cache / "pip-cache")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    pip = [sys.executable, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"], env=env)
    run(
        pip + ["install", *TORCH_STACK, "--index-url", "https://download.pytorch.org/whl/cu128"],
        env=env,
    )

    if not (layout.comfy / ".git").is_dir():
        safe_remove(layout.comfy, layout.content)
        run(["git", "clone", "--filter=blob:none", "https://github.com/comfyanonymous/ComfyUI.git", str(layout.comfy)])
    run(["git", "fetch", "--depth", "1", "origin", COMFYUI_REF], cwd=layout.comfy)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=layout.comfy)

    run(pip + ["install", "-r", str(layout.comfy / "requirements.txt")], env=env)
    run(pip + ["install", *FRONTEND_STACK], env=env)
    run(pip + ["install", "-r", str(source / "requirements.txt"), "modelscope"], env=env)

    run(
        [
            sys.executable,
            str(source / "scripts" / "install_director_private.py"),
            "--source",
            str(source / "vendor_source" / "custom_nodes"),
            "--comfy-root",
            str(layout.comfy),
        ],
        env=env,
    )
    director_requirements = layout.comfy / "custom_nodes" / "ComfyUI-H3-Director" / "requirements.txt"
    if director_requirements.is_file():
        run(pip + ["install", "-r", str(director_requirements)], env=env)

    kj = layout.comfy / "custom_nodes" / "ComfyUI-KJNodes"
    if not (kj / ".git").is_dir():
        safe_remove(kj, layout.content)
        run(["git", "clone", "--filter=blob:none", "https://github.com/kijai/ComfyUI-KJNodes.git", str(kj)])
    if (kj / "requirements.txt").is_file():
        run(pip + ["install", "-r", str(kj / "requirements.txt")], env=env)
    if not skip_sage:
        run(pip + ["install", "sageattention"], env=env)
    return env


def link_model_store(layout: Layout) -> None:
    model_root = layout.comfy / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FOLDERS:
        link = model_root / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            safe_remove(link, layout.comfy)
        link.symlink_to(layout.models / name, target_is_directory=True)
    staging = model_root / ".modelscope-minimax-h3"
    if staging.is_symlink():
        staging.unlink()
    elif staging.exists():
        safe_remove(staging, layout.comfy)
    staging.symlink_to(layout.cache / "modelscope-staging", target_is_directory=True)


def materialize_latent_upscaler(layout: Layout) -> None:
    name = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    cached = layout.models / "latent_upscale_models" / name
    target_dir = layout.comfy / "models" / "latent_upscale_models"
    if not cached.is_file():
        raise FileNotFoundError(cached)
    if target_dir.is_symlink():
        target_dir.unlink()
    elif target_dir.exists():
        safe_remove(target_dir, layout.comfy)
    target_dir.mkdir(parents=True)
    shutil.copy2(cached, target_dir / name)
    print("3D latent upscaler 已复制到本地模型根目录，满足 Director realpath 校验。")


def write_colab_config(layout: Layout, source: Path) -> Path:
    import yaml

    base = yaml.safe_load((source / "config" / "production.5090.cn.yaml").read_text(encoding="utf-8"))
    base["paths"] = {
        "jobs": str(layout.drive / "jobs"),
        "work": str(layout.drive / "work"),
        "review_ready": str(layout.drive / "review_ready"),
        "rejected": str(layout.drive / "rejected"),
    }
    base.setdefault("delivery", {})["directory"] = str(layout.drive / "deliveries")
    base.setdefault("cost", {})["gpu_hourly_cny"] = 0
    # Colab host RAM varies.  A smaller batch/chunk is safer than the 64 GB server baseline.
    base.setdefault("upscale", {})["frame_batch_size"] = 1
    base["upscale"]["chunk_seconds"] = 10
    target = source / "config" / "production.colab.yaml"
    target.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def plan(layout: Layout, profile: str) -> dict[str, object]:
    return {
        "drive_root": str(layout.drive),
        "content_root": str(layout.content),
        "source_repo": SOURCE_REPO,
        "source_cache": str(layout.cache / "private-source-main.zip"),
        "comfyui_ref": COMFYUI_REF,
        "profile": profile,
        "persistent": ["models", "runtime-cache", "jobs", "work", "review_ready", "rejected", "deliveries"],
        "local_runtime": [str(layout.source), str(layout.comfy)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MiniMax H3 for Google Colab + Drive")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/minimaxH3_colab")
    parser.add_argument("--content-root", default="/content")
    parser.add_argument("--profile", choices=("ref2va", "fl2va", "both"), default="ref2va")
    parser.add_argument("--plan", action="store_true", help="print actions without changing anything")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-sage", action="store_true")
    parser.add_argument("--allow-low-vram", action="store_true")
    args = parser.parse_args()
    layout = Layout(Path(args.drive_root).expanduser(), Path(args.content_root).expanduser())
    if args.plan:
        print(json.dumps(plan(layout, args.profile), ensure_ascii=False, indent=2))
        return 0

    ensure_layout(layout)
    gpu = gpu_report()
    ram = host_ram_gib()
    print(json.dumps({"gpu": gpu, "host_ram_gib": ram}, ensure_ascii=False, indent=2))
    if not gpu["name"]:
        raise RuntimeError("当前 Colab 没有 NVIDIA GPU；请在运行时设置中选择 GPU。")
    if float(gpu["vram_gib"]) < 23 and not args.allow_low_vram:
        raise RuntimeError("完整双阶段工作流需要约 24 GB 显存；如明确要试低显存卡，请传 --allow-low-vram。")
    if ram and ram < 48:
        print("警告：宿主内存低于 48 GiB，15 秒任务和 RealESRGAN 超分可能 OOM。")

    source = download_source(layout, os.environ.get("H3_SOURCE_GITHUB_TOKEN"))
    env = install_python_stack(layout, source, args.skip_sage)
    link_model_store(layout)
    if not args.skip_models:
        run(
            [
                sys.executable,
                str(source / "scripts" / "download_models_cn.py"),
                "--comfy-root",
                str(layout.comfy),
                "--profile",
                args.profile,
                "--no-local-library",
            ],
            env=env,
        )
        materialize_latent_upscaler(layout)
    config = write_colab_config(layout, source)
    if not args.skip_models:
        run(
            [
                sys.executable,
                str(source / "scripts" / "verify_runtime.py"),
                "--comfy-root",
                str(layout.comfy),
                "--profile",
                "ref2va" if args.profile == "both" else args.profile,
            ],
            env=env,
        )
    print(json.dumps({"status": "ready", "source": str(source), "comfy": str(layout.comfy), "config": str(config)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


