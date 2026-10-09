# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare stated signatures with the snapshot; this does not grade semantics."""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

from resources_servers.deepcad.knowledge_eval.probe import api_name


def plain(text: str) -> str:
    """Remove Sphinx Markdown formatting while preserving Python star arguments."""
    text = re.sub(r"(?<!\\)\[([^\[\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace(r"\*", "__STAR__").replace("*", "").replace("__STAR__", "*")
    return re.sub(r"\\(.)", r"\1", text)


def parameter_record(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict]:
    """Ignore annotations but retain parameter order, kind and default values."""
    records = []
    positional = node.args.posonlyargs + node.args.args
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        if argument.arg not in ("self", "cls"):
            records.append(
                {
                    "name": argument.arg,
                    "kind": "positional_only" if index < len(node.args.posonlyargs) else "positional",
                    "default": ast.unparse(default) if default is not None else None,
                }
            )
    if node.args.vararg:
        records.append({"name": node.args.vararg.arg, "kind": "vararg", "default": None})
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        records.append(
            {"name": argument.arg, "kind": "keyword_only", "default": ast.unparse(default) if default else None}
        )
    if node.args.kwarg:
        records.append({"name": node.args.kwarg.arg, "kind": "kwarg", "default": None})
    return records


def parse_line(line: str) -> list[dict] | None:
    """Parse one displayed signature without importing or executing model code."""
    line = line.strip().strip("`")
    match = re.match(r"(?:(?:def|class|classmethod|staticmethod) )?[\w.]+\((.*)\)(?:\s*(?:→|->).*)?(?::.*)?$", line)
    if not match:
        return None
    try:
        node = ast.parse("def api(" + match[1] + "):\n pass").body[0]
        return parameter_record(node)
    except SyntaxError:
        return None


def documented_signatures(text: str) -> list[list[dict]]:
    """Read signature lines before the prose, skipping unparseable doc formats."""
    signatures = []
    for line in text.split("</a>\n", 1)[-1].splitlines():
        if not line.strip():
            continue
        line = plain(line)
        if not re.match(r"(?:(?:class|classmethod|staticmethod) )?[\w.]+\(", line):
            break
        record = parse_line(line)
        if record is not None:
            signatures.append(record)
    return signatures


def stated_signatures(answer: str, name: str) -> list[list[dict]]:
    """Inspect definitions and signature text, avoiding ordinary example calls."""
    names = {name.rsplit(".", 1)[-1], "__init__"}
    records = []
    section = re.search(r"(?im)^#{1,4}\s*(?:\d+[.)]?\s*)?Signature[^\n]*\n", answer)
    excerpt = answer[section.end() :] if section else answer
    if section:
        excerpt = re.split(r"(?m)^#{1,4}\s", excerpt, maxsplit=1)[0]
    for block in re.findall(r"```(?:python|py)?\n(.*?)```", excerpt, re.DOTALL):
        compact = " ".join(line.strip() for line in block.splitlines() if not line.lstrip().startswith(("@", "#")))
        match = re.match(r"(?:(?:def|class|classmethod|staticmethod) )?([\w.]+)\(", compact)
        if match and match[1].rsplit(".", 1)[-1] in names:
            record = parse_line(compact)
            if record is not None and record not in records:
                records.append(record)
        try:
            module = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                records.append(parameter_record(node))
    for line in excerpt.splitlines():
        cleaned = line.strip().strip("`")
        match = re.match(r"(?:(?:def|class|classmethod|staticmethod) )?([\w.]+)\(", cleaned)
        if match and match[1].rsplit(".", 1)[-1] in names:
            record = parse_line(cleaned)
            if record is not None and record not in records:
                records.append(record)
    return records


def main() -> None:
    """Write auditable comparisons, with explicit denominators and exclusions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    counts = defaultdict(lambda: defaultdict(int))
    for path in sorted((args.docs / "api").glob("*.md")):
        name = api_name(path)
        raw_path = args.responses / path.with_suffix(".json").name
        if not raw_path.exists():
            rows.append({"api": name, "status": "missing_response"})
            continue
        raw = json.loads(raw_path.read_text())
        choice = raw["response"]["choices"][0]
        answer = choice["message"].get("content") or ""
        expected = documented_signatures(path.read_text())
        predicted = stated_signatures(answer, name)
        names_match = any([p["name"] for p in a] == [p["name"] for p in b] for a in expected for b in predicted)
        exact_match = any(a == b for a in expected for b in predicted)
        group = ".".join(name.split(".")[: 3 if name.startswith("cadquery.occ_impl.") else 2])
        row = {
            "api": name,
            "file": path.name,
            "group": group,
            "finish_reason": choice["finish_reason"],
            "documented_signatures": expected,
            "stated_signatures": predicted,
            "parameter_names_match": names_match,
            "parameter_names_kinds_defaults_match": exact_match,
            "status": "compared" if expected else "reference_signature_unparsed_or_noncallable",
        }
        rows.append(row)
        for key in ("all", group):
            counts[key]["responses"] += 1
            counts[key]["reference_signature_parsed"] += bool(expected)
            counts[key]["model_signature_parsed"] += bool(predicted)
            counts[key]["parameter_names_match"] += names_match
            counts[key]["parameter_names_kinds_defaults_match"] += exact_match
    report = {
        "scope": "Static comparison of explicit signatures, excluding annotations and return types. No code executed.",
        "limitations": [
            "Unparsed model signatures count as no match, including correct answers in unsupported formats.",
            "An overload matching any documented overload counts once; other overloads may still be missing.",
            "Strict default-expression comparison can reject equivalent Python values (for example 0 versus 0.0).",
            "This measures signature recall, not purpose, behavioral accuracy or working CAD generation.",
            "API paths can alias the same implementation; the 808 paths are not 808 independent capabilities.",
        ],
        "counts": dict(counts),
    }
    (args.responses / "signature-comparisons.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.responses / "signature-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
