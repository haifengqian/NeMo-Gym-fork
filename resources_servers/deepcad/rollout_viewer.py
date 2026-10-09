# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve saved DeepCAD rollouts on an automatically selected, persistent port."""

import argparse
import fcntl
import hashlib
import json
import mimetypes
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, unquote, urlparse

from resources_servers.deepcad.live_rollouts import write_json


VERIFICATION_IMAGES = (
    "reference-mask.png",
    "prediction-mask.png",
    "mask-overlap.png",
    "reference-normals.png",
    "prediction-normals.png",
    "reference-holes.png",
    "prediction-holes.png",
)


def contained_path(root: Path, *parts: str) -> Path:
    """Keep all viewer reads inside the configured rollout directory."""
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path outside rollout root")
    return path


class RolloutHandler(BaseHTTPRequestHandler):
    root: Path
    indexes: dict[str, tuple[int, list[dict]]]
    index_lock = Lock()

    def read_index(self, experiment: str) -> list[dict]:
        """Read only newly appended summaries, leaving partial lines for the next poll."""
        path = contained_path(self.root, experiment, "index.jsonl")
        if not path.exists():
            return []
        with self.index_lock:
            offset, rows = self.indexes.get(experiment, (0, []))
            if path.stat().st_size < offset:
                offset, rows = 0, []
            with path.open("rb") as index:
                index.seek(offset)
                added = index.read()
            end = added.rfind(b"\n") + 1
            rows.extend(json.loads(line) for line in added[:end].splitlines() if line)
            self.indexes[experiment] = (offset + end, rows)
            return list(rows)

    def do_GET(self) -> None:
        """Serve experiment pages, saved prompt images, responses and STEP downloads."""
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)
        try:
            if parts == ["api", "health"]:
                self.reply({"service": "deepcad-rollouts", "root": str(self.root)})
            elif parts == ["api", "experiments"]:
                names = sorted(
                    path.name for path in self.root.iterdir() if path.is_dir() and not path.name.startswith(".")
                )
                self.reply({"experiments": names})
            elif len(parts) == 3 and parts[:2] == ["api", "rollouts"]:
                records = self.read_index(parts[2])
                search = query.get("q", [""])[0].lower()
                filtered = [row for row in records if search in (row["task_id"] + row["id"]).lower()]
                order = query.get("sort", ["newest"])[0]
                key = "created_at" if order == "newest" else "reward"
                filtered.sort(key=lambda row: (row[key], row["id"]), reverse=order != "lowest")
                page = max(0, int(query.get("page", ["0"])[0]))
                self.reply(
                    {"total": len(records), "matched": len(filtered), "rows": filtered[page * 50 : (page + 1) * 50]}
                )
            elif len(parts) == 4 and parts[:2] == ["api", "rollout"]:
                experiment, identifier = parts[2:]
                path = contained_path(self.root, experiment, "rollouts", identifier, "reward.json")
                record = json.loads(path.read_text())
                inputs = contained_path(self.root, experiment, "inputs", record["input_hash"] + ".json")
                record["input"] = json.loads(inputs.read_text())
                record["has_step"] = path.with_name("prediction.step").is_file()
                record["has_preview"] = path.with_name("prediction.png").is_file()
                record["verification_images"] = [
                    name for name in VERIFICATION_IMAGES if path.with_name(name).is_file()
                ]
                status = path.with_name("preview-status.json")
                record["preview_status"] = json.loads(status.read_text()) if status.exists() else {"state": "queued"}
                if record["has_step"] and not record["has_preview"] and record["preview_status"]["state"] != "failed":
                    identifier_hash = hashlib.sha256(f"{experiment}/{identifier}".encode()).hexdigest()
                    request = self.root / ".preview-requests" / (identifier_hash + ".json")
                    if not request.exists():
                        write_json(request, {"experiment": experiment, "id": identifier})
                self.reply(record)
            elif (
                len(parts) == 4
                and parts[0] == "file"
                and parts[3] in ("prediction.step", "prediction.png", *VERIFICATION_IMAGES)
            ):
                path = contained_path(self.root, parts[1], "rollouts", parts[2], parts[3])
                self.reply(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            elif not parts or (len(parts) == 1 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", parts[0])):
                self.reply(Path(__file__).with_name("rollout_viewer.html").read_bytes(), "text/html; charset=utf-8")
            else:
                self.send_error(404)
        except (OSError, ValueError, KeyError) as exc:
            self.send_error(404, str(exc))

    def reply(self, value: object, content_type: str = "application/json") -> None:
        payload = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=None, help="Reuse the saved port by default; 0 chooses a free port"
    )
    parser.add_argument("--advertise-host", default=socket.getfqdn())
    parser.add_argument("--experiment", action="append", default=[])
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in args.experiment:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            parser.error("Experiment names must be single directory names")
        (root / name).mkdir(exist_ok=True)
    state = root / ".viewer-state.json"
    with (root / ".viewer.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"already_running": True, **json.loads(state.read_text())}), flush=True)
            return
        preferred = args.port
        if preferred is None:
            preferred = json.loads(state.read_text())["port"] if state.exists() else 0
        handler = type("DeepCADRolloutHandler", (RolloutHandler,), {"root": root, "indexes": {}})
        try:
            server = ThreadingHTTPServer((args.host, preferred), handler)
        except OSError:
            if not preferred:
                raise
            server = ThreadingHTTPServer((args.host, 0), handler)
        info = {"host": args.advertise_host, "port": server.server_port, "root": str(root)}
        write_json(state, info)
        print(json.dumps({"url": f"http://{info['host']}:{info['port']}/", **info}), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
