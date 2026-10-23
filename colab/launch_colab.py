#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Start ComfyUI and the H3 queue worker in Colab")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/minimaxH3_colab")
    parser.add_argument("--content-root", default="/content")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--no-worker", action="store_true")
    args = parser.parse_args()

    content = Path(args.content_root)
    drive = Path(args.drive_root)
    source = content / "comfyui_quick_workflow"
    comfy = content / "ComfyUI"
    runtime_python = content / "h3env" / "bin" / "python"
    config = source / "config" / "production.colab.yaml"
    for required in (runtime_python, source / "src" / "worker.py", comfy / "main.py", config):
        if not required.exists():
            raise FileNotFoundError(f"运行环境未完成，请先执行 bootstrap: {required}")

    logs = drive / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["H3_CONFIG"] = str(config)
    env["PYTHONUNBUFFERED"] = "1"

    base_url = f"http://127.0.0.1:{args.port}"
    pids: dict[str, int] = {}
    if not healthy(base_url + "/system_stats"):
        comfy_log = (logs / "comfyui.log").open("ab", buffering=0)
        process = subprocess.Popen(
            [str(runtime_python), "main.py", "--listen", "127.0.0.1", "--port", str(args.port)],
            cwd=comfy,
            env=env,
            stdout=comfy_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pids["comfyui"] = process.pid
        for _ in range(180):
            if healthy(base_url + "/system_stats"):
                break
            if process.poll() is not None:
                raise RuntimeError(f"ComfyUI 启动失败，请查看 {logs / 'comfyui.log'}")
            time.sleep(2)
        else:
            raise TimeoutError(f"ComfyUI 启动超时，请查看 {logs / 'comfyui.log'}")

    if not args.no_worker:
        worker_log = (logs / "worker.log").open("ab", buffering=0)
        worker = subprocess.Popen(
            [str(runtime_python), "-m", "src.worker", "--config", str(config)],
            cwd=source,
            env=env,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(2)
        if worker.poll() is not None:
            raise RuntimeError(f"worker 启动失败，请查看 {logs / 'worker.log'}")
        pids["worker"] = worker.pid

    state = {
        "status": "running",
        "base_url": base_url,
        "pids": pids,
        "runtime_python": str(runtime_python),
        "config": str(config),
    }
    (content / "h3-colab-runtime.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




