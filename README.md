# MiniMax H3 Colab Launcher

这是 `comfyui_quick_workflow` 的 Google Colab 启动层。成熟工作流本身保持私有；本仓库只负责：

- 挂载 `Google Drive/MyDrive/minimaxH3_colab`
- 从私有仓库取得运行源码，但不把 Token 写入 Drive 或日志
- 把模型、下载缓存、任务、参考图和成片持久化到 Drive
- 每次会话在 `/content` 本地 SSD 重建 ComfyUI 运行目录
- 启动 ComfyUI 与单 GPU 队列 worker

[在 Google Colab 中打开](https://colab.research.google.com/github/SydneyMCDonaldbigking/minimaxH3_colab/blob/main/MinimaxH3_Colab.ipynb)

## 第一次使用

1. 在 Colab 的“密钥”面板添加 `GITHUB_TOKEN`。它只需要读取私有仓库
   `SydneyMCDonaldbigking/comfyui_quick_workflow` 的权限。
2. 把 Colab 运行时切换到 GPU；完整双阶段流程建议至少约 24GB 显存。
3. 按顺序运行 Notebook。首次运行会下载约 43GB 模型到 Drive，因此时间较长。
4. 以后启动会复用 Drive 中的模型、私有源包和 pip 下载缓存。

如果以后不想继续使用 GitHub Token，可以在第一次成功后删除 Colab Secret；启动器会复用
`runtime-cache/private-source-main.zip`。只有需要同步私有仓库的新版本时才重新添加 Token。

## 存储布局

```text
MyDrive/minimaxH3_colab/
├── models/
├── runtime-cache/
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

## 验证启动计划

这条命令不会下载模型或安装依赖：

```bash
python colab/bootstrap_colab.py --plan --profile ref2va
python colab/launch_colab.py --help
```

生产默认仍采用原仓库验证过的 `ref2va_4step`、15 秒、两阶段 latent 修复和
RealESRGAN 2×；Colab 配置只把超分批次降到 1、分块降到 10 秒，以适应波动较大的宿主内存。


