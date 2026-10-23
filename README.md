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

1. 私有源包已经预置在 `runtime-cache/private-source-main.zip`，第一次运行不需要 GitHub Token。
2. 首次准备使用免费的 CPU 运行时；它会安装基础环境、下载约 43GB 模型，并创建环境快照。
3. 真正生成视频时再切到 GPU（完整双阶段流程建议至少约 24GB 显存）。当前 Colab G4/Blackwell 使用 PyTorch 2.11 + CUDA 13.0；Blackwell 默认使用原生 PyTorch attention，避免误装不兼容的通用 SageAttention 包。
4. 以后启动只需挂载 Drive、解压兼容环境快照并校验模型；通常是数分钟级，取决于 Drive 速度和快照大小，不会再重装整套依赖。

只有以后需要从私有仓库同步新版本时，才在 Colab“密钥”面板临时添加
`GITHUB_TOKEN`；它只需要读取 `SydneyMCDonaldbigking/comfyui_quick_workflow` 的权限。

## 存储布局

```text
MyDrive/minimaxH3_colab/
├── models/
├── runtime-cache/
│   ├── private-source-main.zip
│   └── runtime-py*-torch2110-cu130-comfy0340-s2.tar.gz
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
python colab/bootstrap_colab.py --profile ref2va --rebuild-runtime
```

## 验证启动计划

这条命令不会下载模型或安装依赖：

```bash
python colab/bootstrap_colab.py --plan --profile ref2va
python colab/launch_colab.py --help
```

生产默认仍采用原仓库验证过的 `ref2va_4step`、15 秒、两阶段 latent 修复和
RealESRGAN 2×；Colab 配置只把超分批次降到 1、分块降到 10 秒，以适应波动较大的宿主内存。




