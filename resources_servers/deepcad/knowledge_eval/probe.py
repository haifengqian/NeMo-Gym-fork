# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe each documented API independently, exposing only its name to the model."""

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import aiohttp


def api_name(path: Path) -> str:
    """Read the exact symbol, including API entries whose filenames have hashes."""
    match = re.search(r'^title: "(.+)"$', path.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"No API title in {path}")
    return match.group(1)


def question(name: str) -> str:
    """Use the same closed-book question for every symbol, without doc hints."""
    return (
        f"CadQuery API item: {name}\n\n"
        "From your existing knowledge only, explain what this item does. Give its Python signature(s), "
        "including parameter names, defaults, and return type; describe important behavior and constraints; "
        "and provide a small usage example. If it is a class, give its constructor signature. If it is "
        "not callable, explain its type instead of inventing a signature. State explicitly which details "
        "you do not know or are uncertain about. Do not browse, use tools, or assume access to documentation. "
        "Use concise Markdown with sections Purpose, Signature, Behavior, Example, and Uncertainty."
    )


async def run(args: argparse.Namespace) -> None:
    """Save raw JSON and Markdown for every API, resuming completed responses."""
    paths = sorted((args.docs / "api").glob("*.md"))
    if not paths:
        raise ValueError(f"No API documents under {args.docs / 'api'}")
    if args.limit:
        paths = paths[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "base_url": args.base_url,
        "docs": str(args.docs),
        "temperature": 0,
        "seed": 42,
        "prompt_sha256": hashlib.sha256(question("{API_NAME}").encode()).hexdigest(),
        "max_tokens": args.max_tokens,
        "api_count": len(paths),
        "protocol": "One fresh conversation per API. Only the symbol name and a fixed question are supplied.",
        "entries": [
            {"file": p.name, "name": api_name(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths
        ],
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("model", "temperature", "seed", "prompt_sha256", "max_tokens", "entries"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Existing output uses a different {key}; choose a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = 0
    errors = []
    started = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def collect(path: Path) -> None:
            nonlocal completed
            name = api_name(path)
            prompt = question(name)
            raw_path = args.output / path.with_suffix(".json").name
            md_path = args.output / path.name
            if raw_path.exists() and md_path.exists():
                saved = json.loads(raw_path.read_text())
                if saved.get("response", {}).get("choices", [{}])[0].get("finish_reason") == "stop":
                    completed += 1
                    return
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "seed": 42,
                "max_tokens": args.max_tokens,
            }
            async with semaphore:
                for attempt in range(3):
                    try:
                        async with session.post(
                            f"{args.base_url.rstrip('/')}/chat/completions", json=body
                        ) as response:
                            response.raise_for_status()
                            result = await response.json()
                        choice = result["choices"][0]
                        message = choice["message"]
                        answer = message.get("content") or ""
                        if choice["finish_reason"] == "length" and attempt < 2:
                            body["max_tokens"] = min(body["max_tokens"] * 2, 28000)
                            continue
                        record = {
                            "api": name,
                            "reference_file": path.name,
                            "request": body,
                            "response": result,
                        }
                        temporary = raw_path.with_suffix(".json.tmp")
                        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                        temporary.replace(raw_path)
                        reasoning = message.get("reasoning_content") or message.get("reasoning")
                        md = (
                            f"# {name}\n\n"
                            f"Model: `{args.model}`\n\n"
                            f"Finish reason: `{choice['finish_reason']}`\n\n"
                            f"## Prompt\n\n{prompt}\n\n"
                            f"## Model response\n\n{answer}\n"
                        )
                        if reasoning:
                            md += f"\n## Raw reasoning (excluded from answer assessment)\n\n{reasoning}\n"
                        md_path.write_text(md)
                        if choice["finish_reason"] != "stop" or not answer.strip():
                            errors.append(
                                {"api": name, "reason": choice["finish_reason"], "empty": not answer.strip()}
                            )
                        break
                    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as exc:
                        if attempt == 2:
                            errors.append({"api": name, "reason": str(exc)})
                        else:
                            await asyncio.sleep(2 ** (attempt + 1))
            completed += 1
            if completed % 10 == 0 or completed == len(paths):
                print(
                    f"Completed {completed}/{len(paths)} in {time.monotonic() - started:.0f}s; errors={len(errors)}",
                    flush=True,
                )

        await asyncio.gather(*(collect(path) for path in paths))
    (args.output / "collection.json").write_text(
        json.dumps(
            {"completed": completed, "total": len(paths), "errors": errors, "seconds": time.monotonic() - started},
            indent=2,
        )
        + "\n"
    )
    if errors:
        raise RuntimeError(f"{len(errors)} responses need inspection; see collection.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="cadquery-knowledge-step30")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    asyncio.run(run(parser.parse_args()))
