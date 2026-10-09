# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a balanced difficulty band from an already-ranked DeepCAD train split."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cadquery as cq

from resources_servers.deepcad.dataset.prepare_gym import (
    construction_metrics,
    dimension_data,
    prepare_views,
    reconstruct,
    task_row,
)


def prepare_score(
    arguments: tuple[int, list[dict], Path, Path, int, int, float],
) -> tuple[list[dict], list[dict]]:
    """Try a deterministic permutation until this score has enough valid targets."""
    score, candidates, data_root, reference_root, count, seed, probability = arguments
    candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['task_id']}".encode()).digest())
    rows, failures = [], []
    for candidate in candidates:
        identifier = candidate["task_id"]
        try:
            document = json.loads((data_root / "cad_json" / f"{identifier}.json").read_text())
            metrics = construction_metrics(document)
            if metrics != {key: value for key, value in candidate.items() if key != "task_id"}:
                raise ValueError("Source history no longer matches ranked construction metrics")
            shape = reconstruct(document)
            name = identifier.replace("/", "_") + ".step"
            cq.exporters.export(shape, str(reference_root / name))
            shape = cq.importers.importStep(str(reference_root / name)).val()
            if not shape.isValid() or not shape.Solids() or shape.Volume() <= 0:
                raise ValueError("Exported reference failed STEP round-trip solid validation")
            images = prepare_views(
                shape,
                reference_root / "views" / identifier,
                task_id=identifier,
                diagonal_view_probability=probability,
                seed=seed,
                document=document,
            )
            rows.append(
                task_row(
                    identifier, name, images, metrics=metrics, dimensions_mm=dimension_data(shape)["size_mm"].tolist()
                )
            )
        except Exception as exc:  # noqa: BLE001 - retain per-target reconstruction failures
            failures.append({"task_id": identifier, "difficulty": score, "reason": str(exc)})
            continue
        if len(rows) % 10 == 0:
            print(
                json.dumps({"score": score, "prepared": len(rows), "required": count}),
                flush=True,
            )
        if len(rows) == count:
            break
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "ranking", "reference-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--min-score", type=int, default=9)
    parser.add_argument("--max-score", type=int, default=16)
    parser.add_argument("--per-score", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagonal-view-probability", type=float, default=0.75)
    args = parser.parse_args()
    if args.min_score > args.max_score or args.per_score <= 0 or args.workers <= 0:
        parser.error("Require min-score <= max-score, positive per-score and workers")
    if not 0 <= args.diagonal_view_probability <= 1:
        parser.error("--diagonal-view-probability must be between 0 and 1")
    train_ids = set(json.loads((args.data_root / "train_val_test_split.json").read_text())["train"])
    grouped = {score: [] for score in range(args.min_score, args.max_score + 1)}
    for line in args.ranking.read_text().splitlines():
        row = json.loads(line)
        if row["difficulty"] in grouped:
            if row["task_id"] not in train_ids:
                raise ValueError(f"Non-training ID in ranking: {row['task_id']}")
            grouped[row["difficulty"]].append(row)
    args.reference_root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        (
            score,
            candidates,
            args.data_root,
            args.reference_root,
            args.per_score,
            args.seed,
            args.diagonal_view_probability,
        )
        for score, candidates in grouped.items()
    ]
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for prepared, failed in pool.map(prepare_score, arguments):
            rows.extend(prepared)
            failures.extend(failed)
    rows.sort(
        key=lambda row: (
            row["verifier_metadata"]["difficulty"],
            row["verifier_metadata"]["task_id"],
        )
    )
    counts = Counter(row["verifier_metadata"]["difficulty"] for row in rows)
    report = {
        "split": "train",
        "score_range": [args.min_score, args.max_score],
        "per_score": dict(counts),
        "seed": args.seed,
        "written": len(rows),
        "failures": failures,
        "ordering": "ascending difficulty, then task_id",
        "ranking": str(args.ranking),
        "diagonal_view_probability": args.diagonal_view_probability,
        "view_counts": dict(Counter(row["verifier_metadata"]["view_mode"] for row in rows)),
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    if counts != Counter({score: args.per_score for score in grouped}):
        raise RuntimeError("Not enough valid targets in every score; inspect the report")
    if len({row["verifier_metadata"]["task_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "written": len(rows),
                "per_score": dict(counts),
                "failures": len(failures),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
