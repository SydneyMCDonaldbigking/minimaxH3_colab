"""Validate an API-format ComfyUI prompt with this install's own validate_prompt, without running it.

Run from the ComfyUI root:  python validate_api_prompt.py PROMPT.json REPORT.json

Loads ComfyUI and every custom node the way main.py does, but never starts the
server or the queue, so it is safe on a CPU runtime with no models loaded.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

prompt_path, report_path = (Path(p).resolve() for p in sys.argv[1:3])
sys.path.insert(0, os.getcwd())
sys.argv = ["main.py", "--cpu", "--disable-assets"]

import main  # noqa: E402,F401  ComfyUI's module-level setup: paths, logging, prestartup scripts
import execution  # noqa: E402
import nodes  # noqa: E402
import server  # noqa: E402
from app.assets.manager import default_asset_manager  # noqa: E402


def check() -> None:
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.PromptServer(loop, default_asset_manager())
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))

    classes = nodes.NODE_CLASS_MAPPINGS
    missing = sorted({item["class_type"] for item in prompt.values() if item["class_type"] not in classes})
    # validate_prompt reports ok as soon as one output survives, so list every output node too.
    output_nodes = sorted(key for key, item in prompt.items()
                          if getattr(classes.get(item["class_type"]), "OUTPUT_NODE", False) is True)
    ok, error, outputs, node_errors = loop.run_until_complete(
        execution.validate_prompt("h3check", prompt, None))
    summary = {
        "all_ok": bool(ok and not missing and sorted(outputs) == output_nodes and not node_errors),
        "missing_node_types": missing,
        "outputs_passed": f"{len(outputs)}/{len(output_nodes)}",
        "error": error,
        "nodes_with_errors": {
            node_id: [f"{e['message']}: {e['details']}" for e in item["errors"]]
            for node_id, item in node_errors.items()
        },
    }
    report_path.write_text(json.dumps({"summary": summary, "node_errors": node_errors},
                                      ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


# os._exit: custom nodes may leave non-daemon threads behind, and no server is meant to stay up.
status = 0
try:
    check()
except BaseException:
    import traceback
    traceback.print_exc()
    status = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(status)
