#!/usr/bin/env python3
"""动作迁移一次性驱动：绕过 cc_submit / worker，直接向 ComfyUI 打一条带视频参考的任务。

为什么绕过：导演台节点原生支持 ref_videos（每段最多 3 段、2~15s、<Video N> 引用），
但 src/workflow.py 的 build_generation_prompt 把 "video_refs" 和 "video_ref_modes"
写死成空值，包装层没有透出这条通道。仓库本轮不改，所以这里拿它生成的 prompt 再补一刀。

它只读地 import 私有源码模块（Colab 上由 bootstrap 恢复到 /content/comfyui_quick_workflow），
不写源码任何文件。需要 ComfyUI 在 8188（launch_colab.py --no-worker）。
不走 worker，也就没有机器预审、review_ready 和 delivery 那套；产物直接落到 --out。
--dry-run 只生成补好视频参考的 API prompt，不连 ComfyUI，可以在 CPU 上配合 validate_prompt 检查。

用法（Colab，用 bootstrap 装好的 h3env）：
  /content/h3env/bin/python colab/motion_transfer.py \
      --image /content/drive/MyDrive/minimaxH3_colab/assets/motion/<id>/character.png \
      --video /content/drive/MyDrive/minimaxH3_colab/assets/motion/<id>/motion_small.mp4 \
      --prompt-file /content/drive/MyDrive/minimaxH3_colab/assets/motion/<id>/prompt.txt \
      --job-id dance-motion-001 --duration 10 --out /content/h3-motion/out.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("H3_SOURCE_ROOT", "/content/comfyui_quick_workflow"))
sys.path.insert(0, str(ROOT))

from src.comfy_client import ComfyClient
from src.config import load_config, merged_job_config
from src.workflow import build_generation_prompt, output_items, safe_project_id

# 节点的 VIDEO_REFERENCE_MODES：action=只跟身体动作与走位，camera=只跟运镜，
# rhythm=只跟节奏，comprehensive=运镜+动作+节奏全跟。纯动作迁移用 action。
MODES = ("action", "camera", "rhythm", "comprehensive")


def main() -> int:
    parser = argparse.ArgumentParser(description="H3 Ref2VA 动作迁移（图定外观 + 视频定动作）")
    parser.add_argument("--config", default=str(ROOT / "config/production.colab.yaml"))
    parser.add_argument("--image", required=True, action="append",
                        help="外观参考图，可重复；最多 9 张")
    parser.add_argument("--video", action="append", default=[],
                        help="动作参考视频，可重复；最多 3 段，每段 2-15 秒。多分镜用 --slice")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1344)
    parser.add_argument("--video-mode", default="action", choices=MODES)
    parser.add_argument("--video-fps", type=int, default=24,
                        help="参考视频采样帧率；调低=抽帧概括动作、省显存")
    parser.add_argument("--video-skip", type=float, default=0.0, help="跳过参考视频开头秒数")
    parser.add_argument("--no-video-audio", action="store_true",
                        help="不复用参考视频的音轨（默认复用）")
    parser.add_argument("--slice", action="append", default=[], metavar="MP4:SECONDS",
                        help="多分镜模式：每段一片动作视频，格式 路径:时长。"
                             "第 2 段起自动接上一段尾帧（previous_tail），跑完调 merge 合成整片。"
                             "给了 --slice 就忽略 --video/--duration。")
    parser.add_argument("--steps", type=int,
                        help="覆盖采样步数；不给则用 profile 的（ref2va_4step 是 4 步）。"
                             "步数不显著增加显存峰值，只增加时间。")
    parser.add_argument("--only", type=int, metavar="N",
                        help="多分镜：本次只生成第 N 段，其余段禁用但仍占段号。"
                             "62GB 机器上同一进程 stage 两轮必被内核 OOM，"
                             "所以一段一个进程、段间重启 ComfyUI，靠盘上的尾帧接续。")
    parser.add_argument("--merge-only", action="store_true",
                        help="不生成，只把已有各段合并成整片")
    parser.add_argument("--unload-between", action="store_true",
                        help="每段后卸载模型；多分镜默认开启")
    parser.add_argument("--out", required=True, help="产物 MP4 落盘路径")
    parser.add_argument("--dry-run", metavar="PROMPT_JSON",
                        help="不连 ComfyUI、不上传，只把补好视频参考的 API prompt 写到这里")
    args = parser.parse_args()

    if not args.video and not args.slice:
        parser.error("至少要给一个 --video（单段）或 --slice（多分镜）")
    if len(args.image) > 9:
        parser.error("参考图最多 9 张")
    if not args.slice and len(args.video) > 3:
        parser.error("单段模式下参考视频最多 3 段")
    if len(args.image) + len(args.video) > 12:
        parser.error("图 + 视频 + 音频合计不得超过 12 个文件")

    slices = []
    for spec in args.slice:
        path, _, seconds = spec.rpartition(":")
        if not path or not seconds:
            parser.error(f"--slice 格式应为 路径:时长，收到 {spec}")
        slices.append((path, float(seconds)))

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    config = load_config(args.config)
    job = {
        "job_id": args.job_id,
        "prompt": prompt_text,
        "references": [str(Path(p).resolve()) for p in args.image],
        "profile": "ref2va_4step",          # 视频参考只在 Ref2VA 路线生效
        "duration_seconds": args.duration,
        "width": args.width, "height": args.height,
        "second_pass": {"enabled": False},  # 不走二采
        "upscale": {"enabled": False},      # 不走超分
        "metadata": {},
    }
    if slices:
        # 每段一片动作视频；第 2 段起接上一段尾帧，姿势跨接缝连着走。
        # 节点的 _inject_ref2va_tail_reference 会自己把尾帧关系写进 retention_analysis。
        def _shift_pictures(text, by):
            for n in range(9, 0, -1):
                text = text.replace("<Picture %d>" % n, "<Picture %d>" % (n + by))
            return text

        job["segments"] = [
            {"prompt": prompt_text if index == 0 else _shift_pictures(prompt_text, 1),
             "duration_seconds": seconds,
             **({"first_frame_mode": "previous_tail"} if index else {}),
             **({"enabled": args.only == index + 1} if args.only else {})}
            for index, (_, seconds) in enumerate(slices)
        ]
    cfg = merged_job_config(config, job)

    sources = [path for path, _ in slices] if slices else args.video
    if args.dry_run:
        # Same "subfolder/name" shape ComfyClient.upload returns, without touching the server.
        client = None
        uploaded_images = [f"h3jobs/{args.job_id}/{Path(p).name}" for p in job["references"]]
        uploaded_videos = [f"h3jobs/{args.job_id}/{Path(p).name}" for p in sources]
    else:
        comfy = cfg["comfy"]
        client = ComfyClient(comfy["base_url"], int(comfy.get("job_timeout_seconds", 3600)),
                             float(comfy.get("poll_seconds", 3)),
                             int(comfy.get("download_timeout_seconds", 21600)))
        client.health()
        uploaded_images = [client.upload(p, f"h3jobs/{args.job_id}") for p in job["references"]]
        uploaded_videos = [client.upload(p, f"h3jobs/{args.job_id}") for p in sources]
    print(json.dumps({"uploaded_images": uploaded_images,
                      "uploaded_videos": uploaded_videos}, ensure_ascii=False), flush=True)

    # 多分镜时角色图必须挂在每一段自己的 refs 上。顶层共享位（text_shared_refs_json）
    # 是导演台前端的机制，无头调用时段内 refs 为空，日志会打"参考图 0 张"——
    # 005 第一版整条片子人物跑飞就是这么来的。
    prompt = build_generation_prompt(
        job, cfg, uploaded_images, [list(uploaded_images) for _ in slices] if slices else None)

    # ——补上包装层没透出的那一刀——
    node = prompt["13"]["inputs"]
    segments = json.loads(node["segments_json"])
    for index, segment in enumerate(segments):
        # 多分镜：第 N 段只看第 N 片动作视频；单段：所有视频都给这一段。
        segment["video_refs"] = [uploaded_videos[index]] if slices else uploaded_videos
        segment["video_ref_modes"] = {name: args.video_mode for name in segment["video_refs"]}
        segment["video_fps"] = args.video_fps
        segment["video_skip"] = args.video_skip
        # 节点只在这个开关为真时才去解参考视频的音轨（studio_node.py 的
        # _load_video_for_ref）。不设的话日志会打"仅参考画面、不发送原音轨"，
        # 音频复用就是空谈——第一轮就是这么踩的。
        segment["video_audio_reference"] = not args.no_video_audio
    node["segments_json"] = json.dumps(segments, ensure_ascii=False)
    if args.steps:
        node["steps"] = int(args.steps)
    if len(segments) > 1 or args.unload_between:
        # 62GB 宿主机上 H3+Qwen 一轮 staged 就要 40GB，同一进程连跑两轮必被内核 OOM
        # （实测两次，都停在 anon-rss 45GB）。多分镜段间卸载模型把峰值压回单段水平，
        # 代价是每段重新加载约一分钟。
        node["每段后卸载模型"] = True

    if args.dry_run:
        Path(args.dry_run).write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"dry_run": args.dry_run, "segments": len(segments),
                          "video_refs": [s["video_refs"] for s in segments]}, ensure_ascii=False))
        return 0

    started = time.time()
    if args.merge_only:
        project = safe_project_id(args.job_id)
        final = {"filename": client.merge_segments(project, len(slices)),
                 "subfolder": f"video/h3director/{project}", "type": "output"}
        out = Path(args.out)
        client.download(final, out)
        print(json.dumps({"job_id": args.job_id, "merged_segments": len(slices),
                          "result": str(out.resolve()), "size_bytes": out.stat().st_size},
                         ensure_ascii=False, indent=2))
        return 0
    prompt_id, history = client.run(prompt)
    items = output_items(history, "13") or output_items(history)
    if not items:
        raise RuntimeError("生成完成但历史记录中没有 MP4")
    if len(segments) > 1 and not args.only:
        # 各段 MP4 落在 output/video/h3director/<project>/，节点自己不合并。
        # 漏了这一步只会拿到最后一段。project_id 会被导演台改写，必须用 safe_project_id。
        if len(items) < len(segments):
            raise RuntimeError(f"应生成 {len(segments)} 段，历史记录只有 {len(items)} 个 MP4")
        project = safe_project_id(args.job_id)
        final = {"filename": client.merge_segments(project, len(segments)),
                 "subfolder": f"video/h3director/{project}", "type": "output"}
    else:
        final = items[-1]
    out = Path(args.out)
    client.download(final, out)
    print(json.dumps({
        "job_id": args.job_id, "prompt_id": prompt_id,
        "video_refs": uploaded_videos, "video_mode": args.video_mode,
        "elapsed_seconds": round(time.time() - started, 1),
        "estimated_gpu_cost_cny": round((time.time() - started) / 3600
                                        * float(cfg.get("cost", {}).get("gpu_hourly_cny", 0)), 4),
        "result": str(out.resolve()), "size_bytes": out.stat().st_size,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
