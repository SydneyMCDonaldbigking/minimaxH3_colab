#!/usr/bin/env python3
"""Make the private H3 Director call native MiniMaxH3ReferenceToVideo by keyword.

ComfyUI bc5f880 reordered MiniMaxH3ReferenceToVideo.execute to (clip, prompt, width,
height, length, ref_image_size, vae, audio_vae, ...). The Director in the runtime
snapshot still passes (clip, vae, audio_vae, prompt, ...) by position, so every Ref2VA
segment fails with "got multiple values for argument 'ref_image_size'". Keywords
work with both orders. Only the runtime copy under /content is patched; running it
again is a no-op.

Usage: python patch_director.py [path/to/studio_node.py]
"""
import sys
from pathlib import Path

STUDIO = Path("/content/ComfyUI/custom_nodes/ComfyUI-H3-Director/studio_node.py")
OLD = ("            return MiniMaxH3ReferenceToVideo.execute(\n"
       "                clip, vae, audio_vae, condition_prompt, condition_width, condition_height, length,\n")
NEW = ("            return MiniMaxH3ReferenceToVideo.execute(\n"
       "                clip=clip, vae=vae, audio_vae=audio_vae, prompt=condition_prompt,\n"
       "                width=condition_width, height=condition_height, length=length,\n")


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return "already patched"
    if text.count(OLD) != 1:
        raise SystemExit(f"{path} 的导演台源码和预期不一致，未打补丁")
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return "patched"


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else STUDIO
    print(f"{patch(target)}: {target}")
