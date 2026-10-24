# MiniMax H3 Colab Launcher

这是 `comfyui_quick_workflow` 的 Google Colab 启动层。成熟工作流本身保持私有；本仓库只负责：

- 挂载 `Google Drive/MyDrive/minimaxH3_colab`
- 从私有仓库取得运行源码，但不把 Token 写入 Drive 或日志
- 把模型、下载缓存、任务、参考图和成片持久化到 Drive
- 首次构建 `venv + ComfyUI + 私有工作流`，保存成版本化 Drive 环境快照
- 后续会话把兼容快照恢复到 `/content` 本地 SSD，不再重新跑 pip 安装
- 启动 ComfyUI 与单 GPU 队列 worker

[在 Google Colab 中打开](https://colab.research.google.com/github/SydneyMCDonaldbigking/minimaxH3_colab/blob/main/MinimaxH3_Colab.ipynb)

## 第一次使用

1. 新舞蹈工作流首次同步需要 Colab Secret `GITHUB_TOKEN` 读取私有分支；同步后会缓存为 `runtime-cache/private-source-fast-h3-dance-swap.zip`，后续不再需要重复下载。
2. 首次准备使用免费的 CPU 运行时；`dance_swap` 会复用已有 H3 文件，并把 FastH3 OmniRef INT8、Character Swap V1.5 与 SAM 3.1（约 24GB）直接保存到 Drive。
3. 真正生成视频时再切到 GPU（本项目默认要求约 40GB 以上显存）。当前 Colab G4/Blackwell 使用 PyTorch 2.11 + CUDA 13.0；FastH3 固定使用 VSA 10% 与 Comfy Kitchen Attention。
4. 以后启动只需挂载 Drive、解压兼容环境快照并校验模型；通常是数分钟级，取决于 Drive 速度和快照大小，不会再重装整套依赖。

只有以后需要从私有仓库同步新版本时，才在 Colab“密钥”面板临时添加
`GITHUB_TOKEN`；它只需要读取 `SydneyMCDonaldbigking/comfyui_quick_workflow` 的权限。

## 存储布局

```text
MyDrive/minimaxH3_colab/
├── models/
│   ├── checkpoints/sam3.1_multiplex_fp16.safetensors
│   ├── diffusion_models/fasth3_omniref_8step_pruned_int8_convrot.safetensors
│   └── loras/h3_character_swap_v1.5.safetensors
├── runtime-cache/
│   ├── private-source-fast-h3-dance-swap.zip
│   └── runtime-py*-torch2110-cu130-comfybc5f880-s3.tar.gz
├── jobs/{inbox,running,done,failed}/
├── work/
├── assets/
├── private-nodes/
├── review_ready/
├── rejected/
├── deliveries/
└── logs/
```

3D latent upscaler 会持久化在 Drive，但每次启动还会复制一份到本地 ComfyUI 模型目录，
以满足 Director 节点的 `realpath` 安全检查。

只有 ComfyUI、Python 虚拟环境和工作流源码进入环境快照；约 43GB 模型始终单独保存在
`models/`，不会被重复打包。升级 ComfyUI、PyTorch 或私有工作流以后，可显式重建：

```bash
python colab/bootstrap_colab.py --profile dance_swap --rebuild-runtime
```

## 验证启动计划

这条命令不会下载模型或安装依赖：

```bash
python colab/bootstrap_colab.py --plan --profile dance_swap
python colab/launch_colab.py --help
```

`dance_swap` 会安装 `H3-Character-Swap-1.5-SAM3.1-FastH3-OmniRef.json`，保留现成
工作流的 SAM 追踪、动态裁剪、局部回填和视频导出，只把 VDN8 模型链替换为
FastH3 OmniRef 的 8 步 Euler + simple 调度。该画布按设计保留原背景；需要整帧换景时
用下面的整帧动作迁移。

## 整帧动作迁移（保留舞蹈动作，换人换景）

`colab/run_motion_transfer.py` 重新生成整帧画面：外观和场景来自角色图（`<Picture 1>`），
动作来自舞蹈视频（`<Video 1>`，action 模式）。走 Ref2VA 4 步，即
`minimax_h3_ref2va_pruned_int8_convrot` + `minimax_h3_ref2v_turbo_4step_v0.1` LoRA，
`ref2va` profile 已经下载的文件就够，不需要额外模型。

```bash
# GPU 运行时（推荐 G4），Drive 已挂载到 /content/drive
python colab/run_motion_transfer.py \
  --image /content/drive/MyDrive/minimaxH3_colab/assets/motion/<id>/character.png \
  --video /content/drive/MyDrive/minimaxH3_colab/assets/motion/<id>/dance.mp4 \
  --prompt-file /content/drive/MyDrive/minimaxH3_colab/assets/motion/<id>/prompt.txt \
  --job-id <id> --draws 3 --mux-original-audio
```

- `--draws N` 随机抽 N 个 seed、`--seeds 101,102,103` 指定 seed，每个 seed 出一条 `<id>_s<seed>.mp4`。
  同一配置只换 seed 差异就很大，比较效果时每格至少 3 条。宿主机内存不足 80GB 时每条之间重启 ComfyUI。

- 参考视频截到最多 15 秒，缩到长边 960、24fps；成片 768×1344（竖）或 1344×768（横），默认跟参考视频一致。
- 提示词写成六段结构时，`retention_analysis` 必须带所选模式的官方声明原句，否则导演台节点会跳过
  `--video-mode`；脚本发现缺了会直接拒绝。
- 跑完核对 ComfyUI 日志：参考图至少 1 张、出现 `参考视频 <Video 1>`、`已发送原音轨`，缺一条就以非零退出。
- 成片、预处理后的参考视频、提示词和 `run.json` 存到 `MyDrive/minimaxH3_colab/deliveries/motion/<id>/`。
- 加 `--dry-run` 可在 CPU 运行时上只生成 API prompt，并交给 ComfyUI 的 `validate_prompt` 校验，不生成视频；
  检查产物留在 `/content/h3-motion/<id>/check/`，不写进 Drive。




