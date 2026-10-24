#!/usr/bin/env python3
"""Install the private MiniMax H3 workflow into a Colab runtime.

The public launcher never vendors private nodes.  A small archive of the
private source and all large model/cache files live in the user's Drive.
The first run builds a versioned runtime snapshot; later Colab sessions
restore it to the local SSD instead of reinstalling the Python stack.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


SOURCE_REPO = "SydneyMCDonaldbigking/comfyui_quick_workflow"
SOURCE_REF = "codex/fast-h3-dance-swap"
COMFYUI_REF = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
SNAPSHOT_SCHEMA = 3
TORCH_STACK = ("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0")
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
TORCH_SNAPSHOT_TAG = "torch2110-cu130"
FRONTEND_STACK = (
    "comfyui-frontend-package==1.57.1",
    "comfyui-workflow-templates==0.11.96",
    "comfyui-embedded-docs==0.5.14",
    "comfy-kitchen==0.2.40",
    "comfy-aimdo==0.6.0",
)
CUSTOM_NODE_REPOS = (
    (
        "ComfyUI-KJNodes",
        "https://github.com/kijai/ComfyUI-KJNodes.git",
        "37a6c45a07bb06cc392431ddbbc741e691c467f4",
    ),
    (
        "ComfyUI-VideoHelperSuite",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "4d907bee61e92c2e65af3bd6383a4e4d356126d1",
    ),
    (
        "MaskVidExperiments",
        "https://github.com/drozbay/MaskVidExperiments.git",
        "c32ed8c17e6fe892a174ebf25fe98b1317de97fb",
    ),
)
MODEL_FOLDERS = (
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "latent_upscale_models",
    "upscale_models",
    "vae_approx",
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

    @property
    def venv(self) -> Path:
        return self.content / "h3env"

    @property
    def runtime_python(self) -> Path:
        relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
        return self.venv / relative

    @property
    def snapshot_key(self) -> str:
        py = f"py{sys.version_info.major}{sys.version_info.minor}"
        return f"{py}-{TORCH_SNAPSHOT_TAG}-comfy{COMFYUI_REF[:7]}-s{SNAPSHOT_SCHEMA}"

    @property
    def snapshot(self) -> Path:
        return self.cache / f"runtime-{self.snapshot_key}.tar.gz"

    @property
    def snapshot_manifest(self) -> Path:
        return self.cache / f"runtime-{self.snapshot_key}.json"


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
    archive = layout.cache / "private-source-fast-h3-dance-swap.zip"
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


def runtime_env(layout: Layout) -> dict[str, str]:
    env = os.environ.copy()
    local_pip_cache = layout.content / ".cache" / "pip"
    local_pip_cache.mkdir(parents=True, exist_ok=True)
    env["PIP_CACHE_DIR"] = str(local_pip_cache)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["VIRTUAL_ENV"] = str(layout.venv)
    env["PATH"] = str(layout.runtime_python.parent) + os.pathsep + env.get("PATH", "")
    return env


def runtime_ready(layout: Layout) -> bool:
    return all(
        path.is_file()
        for path in (
            layout.runtime_python,
            layout.comfy / "main.py",
            layout.source / "src" / "worker.py",
        )
    )


def clean_local_runtime(layout: Layout) -> None:
    for path in (layout.venv, layout.comfy, layout.source):
        safe_remove(path, layout.content)


def restore_runtime_snapshot(layout: Layout) -> bool:
    if not layout.snapshot.is_file():
        print(f"未找到兼容的环境快照，将执行首次安装: {layout.snapshot.name}")
        return False
    print(f"正在从 Drive 恢复环境快照: {layout.snapshot}")
    started = time.monotonic()
    clean_local_runtime(layout)
    try:
        with tarfile.open(layout.snapshot, "r:gz") as bundle:
            content_root = layout.content.resolve(strict=False)
            for member in bundle.getmembers():
                target = (layout.content / member.name).resolve(strict=False)
                if target != content_root and content_root not in target.parents:
                    raise RuntimeError(f"环境快照含有越界路径: {member.name}")
            bundle.extractall(layout.content)
        if not runtime_ready(layout):
            raise RuntimeError("环境快照不完整")
    except (OSError, tarfile.TarError, RuntimeError) as error:
        print(f"环境快照恢复失败，将自动重建: {error}")
        clean_local_runtime(layout)
        return False
    print(f"环境快照恢复完成，用时 {time.monotonic() - started:.1f} 秒。")
    return True


def create_runtime_snapshot(layout: Layout) -> None:
    if not runtime_ready(layout):
        raise RuntimeError("运行环境未完成，拒绝创建快照")
    partial = layout.snapshot.with_suffix(layout.snapshot.suffix + ".partial")
    partial.unlink(missing_ok=True)
    print(f"正在把可复用环境保存到 Drive（只在首次安装或版本升级时执行）: {layout.snapshot}")
    started = time.monotonic()

    def exclude_large_or_rebuildable(item: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(item.name).parts
        if ".git" in parts:
            return None
        if len(parts) >= 2 and parts[0] == layout.comfy.name and parts[1] == "models":
            return None
        return item

    try:
        with tarfile.open(partial, "w:gz", compresslevel=1) as bundle:
            for path in (layout.venv, layout.comfy, layout.source):
                bundle.add(path, arcname=path.name, recursive=True, filter=exclude_large_or_rebuildable)
        partial.replace(layout.snapshot)
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "key": layout.snapshot_key,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": TORCH_STACK[0],
            "torch_index_url": TORCH_INDEX_URL,
            "comfyui_ref": COMFYUI_REF,
            "archive": layout.snapshot.name,
            "models_included": False,
        }
        layout.snapshot_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        partial.unlink(missing_ok=True)
    print(f"环境快照已保存，用时 {time.monotonic() - started:.1f} 秒。")


def install_python_stack(layout: Layout, source: Path, skip_sage: bool) -> dict[str, str]:
    safe_remove(layout.venv, layout.content)
    try:
        run([sys.executable, "-m", "venv", str(layout.venv)])
    except subprocess.CalledProcessError:
        print("系统 venv/ensurepip 不可用，改用 virtualenv 创建隔离环境。")
        safe_remove(layout.venv, layout.content)
        run([sys.executable, "-m", "pip", "install", "virtualenv"])
        run([sys.executable, "-m", "virtualenv", str(layout.venv)])
    env = runtime_env(layout)
    pip = [str(layout.runtime_python), "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"], env=env)
    run(
        pip + ["install", *TORCH_STACK, "--index-url", TORCH_INDEX_URL],
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
            str(layout.runtime_python),
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

    for name, url, ref in CUSTOM_NODE_REPOS:
        node = layout.comfy / "custom_nodes" / name
        if not (node / ".git").is_dir():
            safe_remove(node, layout.content)
            run(["git", "clone", "--filter=blob:none", url, str(node)])
        run(["git", "fetch", "--depth", "1", "origin", ref], cwd=node)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=node)
        if (node / "requirements.txt").is_file():
            run(pip + ["install", "-r", str(node / "requirements.txt")], env=env)
    return env


def ensure_sage_attention(layout: Layout, env: dict[str, str]) -> bool:
    capability_probe = subprocess.run(
        [
            str(layout.runtime_python),
            "-c",
            (
                "import json, torch; "
                "print(json.dumps(torch.cuda.get_device_capability(0) "
                "if torch.cuda.is_available() else None))"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        capability = json.loads(capability_probe.stdout.strip())
    except json.JSONDecodeError:
        capability = None
    if capability and tuple(capability) >= (12, 0):
        print(
            "检测到 Blackwell GPU (sm_120+)：跳过不兼容的通用 PyPI SageAttention。"
            "当前使用 PyTorch 原生 attention；专用 SageAttention 需匹配 Python、Torch、CUDA "
            "与 sm_120 单独编译。"
        )
        return False

    probe = subprocess.run(
        [str(layout.runtime_python), "-c", "import sageattention"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        return False
    try:
        run([str(layout.runtime_python), "-m", "pip", "install", "sageattention"], env=env)
    except subprocess.CalledProcessError as error:
        print(f"警告：SageAttention 安装失败，继续使用 PyTorch 原生 attention: {error}")
        return False
    return True


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


def install_dance_swap_workflow(layout: Layout, source: Path) -> Path | None:
    name = "H3-Character-Swap-1.5-SAM3.1-FastH3-OmniRef.json"
    workflow = source / "workflows" / name
    if not workflow.is_file():
        print(f"私有源包暂不包含舞蹈换人工作流: {workflow}")
        return None
    destination = layout.comfy / "user" / "default" / "workflows" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workflow, destination)
    print(f"舞蹈换人工作流已安装: {destination}")
    return destination


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
        "source_ref": SOURCE_REF,
        "source_cache": str(layout.cache / "private-source-fast-h3-dance-swap.zip"),
        "runtime_snapshot": str(layout.snapshot),
        "runtime_snapshot_key": layout.snapshot_key,
        "comfyui_ref": COMFYUI_REF,
        "torch_stack": list(TORCH_STACK),
        "torch_index_url": TORCH_INDEX_URL,
        "profile": profile,
        "persistent": ["models", "runtime-cache", "jobs", "work", "review_ready", "rejected", "deliveries"],
        "local_runtime": [str(layout.venv), str(layout.source), str(layout.comfy)],
        "startup_strategy": "restore compatible Drive snapshot; rebuild only when missing or requested",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MiniMax H3 for Google Colab + Drive")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/minimaxH3_colab")
    parser.add_argument("--content-root", default="/content")
    parser.add_argument(
        "--profile", choices=("ref2va", "fl2va", "both", "dance_swap"), default="ref2va"
    )
    parser.add_argument("--plan", action="store_true", help="print actions without changing anything")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-sage", action="store_true")
    parser.add_argument("--allow-low-vram", action="store_true")
    parser.add_argument(
        "--prepare-on-cpu",
        action="store_true",
        help="allow first-time environment/model preparation without reserving a GPU",
    )
    parser.add_argument(
        "--rebuild-runtime",
        action="store_true",
        help="ignore/delete the compatible local runtime and rebuild its Drive snapshot",
    )
    args = parser.parse_args()
    layout = Layout(Path(args.drive_root).expanduser(), Path(args.content_root).expanduser())
    if args.plan:
        print(json.dumps(plan(layout, args.profile), ensure_ascii=False, indent=2))
        return 0

    ensure_layout(layout)
    gpu = gpu_report()
    ram = host_ram_gib()
    print(json.dumps({"gpu": gpu, "host_ram_gib": ram}, ensure_ascii=False, indent=2))
    cpu_prepare = not gpu["name"] and args.prepare_on_cpu
    if not gpu["name"] and not cpu_prepare:
        raise RuntimeError(
            "当前 Colab 没有 NVIDIA GPU；如只做环境和模型准备，请传 --prepare-on-cpu。"
        )
    if gpu["name"] and float(gpu["vram_gib"]) < 23 and not args.allow_low_vram:
        raise RuntimeError("完整双阶段工作流需要约 24 GB 显存；如明确要试低显存卡，请传 --allow-low-vram。")
    if cpu_prepare:
        print("CPU 准备模式：下载模型、构建基础环境和快照；暂不安装 SageAttention，也不做 CUDA 验证。")
    elif ram and ram < 48:
        print("警告：宿主内存低于 48 GiB，15 秒任务和 RealESRGAN 超分可能 OOM。")

    runtime_origin = "local"
    if args.rebuild_runtime:
        clean_local_runtime(layout)
        layout.snapshot.unlink(missing_ok=True)
        layout.snapshot_manifest.unlink(missing_ok=True)
        ready = False
    elif runtime_ready(layout):
        print("检测到本次 Colab 会话中的完整环境，直接复用。")
        ready = True
    else:
        ready = restore_runtime_snapshot(layout)
        if ready:
            runtime_origin = "snapshot"

    if ready:
        source = layout.source
        env = runtime_env(layout)
    else:
        source = download_source(layout, os.environ.get("H3_SOURCE_GITHUB_TOKEN"))
        env = install_python_stack(layout, source, args.skip_sage or cpu_prepare)
        runtime_origin = "built"

    sage_added = False
    if gpu["name"] and not args.skip_sage:
        sage_added = ensure_sage_attention(layout, env)
    link_model_store(layout)
    dance_workflow = install_dance_swap_workflow(layout, source)
    if not args.skip_models:
        run(
            [
                str(layout.runtime_python),
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
    if not args.skip_models and gpu["name"]:
        run(
            [
                str(layout.runtime_python),
                str(source / "scripts" / "verify_runtime.py"),
                "--comfy-root",
                str(layout.comfy),
                "--profile",
                args.profile,
            ],
            env=env,
        )
    elif not args.skip_models:
        print("模型下载完成；CUDA 验证将在首次 GPU 会话中执行。")
    if runtime_origin == "built" or sage_added or not layout.snapshot.is_file():
        create_runtime_snapshot(layout)
    print(
        json.dumps(
            {
                "status": "prepared" if cpu_prepare else "ready",
                "runtime_origin": runtime_origin,
                "cpu_prepare": cpu_prepare,
                "sageattention_added": sage_added,
                "runtime_python": str(layout.runtime_python),
                "runtime_snapshot": str(layout.snapshot),
                "source": str(source),
                "comfy": str(layout.comfy),
                "config": str(config),
                "dance_swap_workflow": str(dance_workflow) if dance_workflow else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




