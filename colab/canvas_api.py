"""Turn a ComfyUI canvas workflow into an API prompt and validate it, without running it.

Run from the ComfyUI root:  python canvas_api.py CANVAS.json API.json REPORT.json INPUT_DIR

Loads ComfyUI and every custom node the way main.py does, but never starts the
server or the queue.  The canvas is turned into an API prompt from the live node
definitions, each node's saved widgets are checked against the current widget
order, and the prompt goes through ComfyUI's own execution.validate_prompt.
A combo value may be given as the leading part of an option ("9:16" for
"9:16 (Portrait)"); it is replaced by the one option it names.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

workflow_path, api_path, report_path, input_dir = (Path(p).resolve() for p in sys.argv[1:5])
sys.path.insert(0, os.getcwd())
sys.argv = ["main.py", "--cpu", "--disable-assets", "--input-directory", str(input_dir)]

import main  # noqa: E402,F401  ComfyUI's module-level setup: paths, logging, prestartup scripts
import execution  # noqa: E402
import nodes  # noqa: E402
import server  # noqa: E402
from app.assets.manager import default_asset_manager  # noqa: E402
from comfy_api.internal import _ComfyNodeInternal  # noqa: E402

UI_ONLY = {"MarkdownNote", "Note", "PrimitiveNode", "Reroute"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}


def node_info(class_type):
    cls = nodes.NODE_CLASS_MAPPINGS[class_type]
    if issubclass(cls, _ComfyNodeInternal):
        return cls.GET_NODE_INFO_V1()
    types = cls.INPUT_TYPES()
    return {"input": types, "input_order": {key: list(value) for key, value in types.items()}}


def spec_of(info, name):
    for group in ("required", "optional"):
        spec = info["input"].get(group, {}).get(name)
        if spec is not None:
            return spec
    return None


def is_widget(spec):
    return spec is not None and (isinstance(spec[0], list) or spec[0] in WIDGET_TYPES)


def has_control(spec):
    return bool(spec and len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("control_after_generate"))


def assign_like_frontend(info, values):
    """Hand widgets_values to widgets in definition order, as the frontend does on load."""
    assigned, stream = {}, iter(values)

    def take(name, spec):
        try:
            assigned[name] = next(stream)
        except StopIteration:
            return False
        if has_control(spec):
            next(stream, None)
        return True

    order = info["input_order"]
    for name in order.get("required", []) + order.get("optional", []):
        spec = spec_of(info, name)
        if not is_widget(spec):
            continue
        if not take(name, spec):
            break
        if spec[0] == "COMFY_DYNAMICCOMBO_V3":
            for option in spec[1].get("options", []):
                if option["key"] == assigned[name]:
                    for group in ("required", "optional"):
                        for sub, sub_spec in option["inputs"].get(group, {}).items():
                            take(f"{name}.{sub}", sub_spec)
    return assigned, list(stream)


def to_api(workflow):
    by_id = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    prompt, widget_checks = {}, []

    def upstream(link_id):
        link = links.get(link_id)
        source = by_id.get(link[1]) if link else None
        if source is None or source.get("mode") == 2 or source["type"] in UI_ONLY - {"PrimitiveNode"}:
            return None
        if source["type"] == "PrimitiveNode":
            return ("value", source["widgets_values"][0])
        if source.get("mode") == 4:  # bypassed: pass the first same-typed input through
            for item in source.get("inputs", []):
                if item.get("link") is not None and item.get("type") == link[5]:
                    return upstream(item["link"])
            return None
        return ("link", [str(source["id"]), link[2]])

    for node in workflow["nodes"]:
        kind = node["type"]
        if kind in UI_ONLY or node.get("mode") in (2, 4) or kind not in nodes.NODE_CLASS_MAPPINGS:
            continue
        info = node_info(kind)
        values = node.get("widgets_values")
        if isinstance(values, dict):
            inputs = {key: value for key, value in values.items() if key != "videopreview"}
            widget_checks.append({"id": node["id"], "type": kind, "ok": True, "how": "按名称保存"})
        else:
            inputs, leftover = assign_like_frontend(info, values or [])
            saved = [item["widget"]["name"] for item in node.get("inputs", []) if item.get("widget")]
            saved = [name for name in saved if spec_of(info, name.split(".")[0]) is not None]
            named = node.get("widgets_values_named")
            if named is not None:
                ok, how = named == inputs, "对照 widgets_values_named"
            elif saved:
                ok, how = saved == list(inputs), "对照画布保存的控件名"
            else:
                ok, how = True, "无控件"
            widget_checks.append({
                "id": node["id"], "type": kind, "ok": ok, "how": how,
                "assigned": inputs, "expected": named if named is not None else saved,
                "leftover": leftover,
            })
        for item in node.get("inputs", []):
            if item.get("link") is None:
                continue
            source = upstream(item["link"])
            if source is None:
                inputs.pop(item["name"], None)
            else:
                inputs[item["name"]] = source[1]
        prompt[str(node["id"])] = {
            "class_type": kind,
            "inputs": inputs,
            "_meta": {"title": node.get("title") or kind},
        }
    return prompt, widget_checks


def resolve_combo_prefixes(prompt):
    """Expand short combo values such as "9:16" to the single option they start."""
    for item in prompt.values():
        info = node_info(item["class_type"])
        for name, value in item["inputs"].items():
            spec = spec_of(info, name)
            if not isinstance(value, str) or spec is None:
                continue
            extra = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            options = spec[0] if isinstance(spec[0], list) else extra.get("options") if spec[0] == "COMBO" else None
            if not options or value in options:
                continue
            matches = [option for option in options if isinstance(option, str) and option.startswith(value + " ")]
            if len(matches) == 1:
                item["inputs"][name] = matches[0]


def main_check():
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.PromptServer(loop, default_asset_manager())
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))

    used = sorted({node["type"] for node in workflow["nodes"] if node["type"] not in UI_ONLY})
    missing = [kind for kind in used if kind not in nodes.NODE_CLASS_MAPPINGS]
    prompt, widget_checks = to_api(workflow)
    resolve_combo_prefixes(prompt)
    api_path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8")
    # validate_prompt reports ok as soon as one output survives, so list every output node too.
    output_nodes = sorted(
        key for key, item in prompt.items()
        if getattr(nodes.NODE_CLASS_MAPPINGS[item["class_type"]], "OUTPUT_NODE", False) is True
    )
    ok, error, outputs, node_errors = loop.run_until_complete(
        execution.validate_prompt("h3check", prompt, None)
    )
    all_ok = (not missing and all(c["ok"] for c in widget_checks) and ok
              and sorted(outputs) == output_nodes and not node_errors)
    report = {
        "all_ok": all_ok,
        "node_types_used": used,
        "missing_node_types": missing,
        "widget_checks": widget_checks,
        "output_nodes": output_nodes,
        "prompt": prompt,
        "validate_prompt": {"ok": ok, "error": error, "outputs": sorted(outputs), "node_errors": node_errors},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "all_ok": all_ok,
        "missing_node_types": missing,
        "widget_order_problems": [c["id"] for c in widget_checks if not c["ok"]],
        "outputs_passed": f"{len(outputs)}/{len(output_nodes)}",
        "nodes_with_errors": sorted(node_errors),
    }, ensure_ascii=False, indent=2, default=str))


# os._exit: custom nodes may leave non-daemon threads behind, and no server is meant to stay up.
status = 0
try:
    main_check()
except BaseException:
    import traceback
    traceback.print_exc()
    status = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(status)
