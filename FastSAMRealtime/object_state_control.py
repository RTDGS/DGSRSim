"""Command-line control for active objects in a DGSRSim state bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.object_state_bundle import MULTI_STATE_SCHEMA, set_object_active


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Activate, deactivate, or inspect DGSRSim objects")
    parser.add_argument("state_json", help="Path to object_states.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("activate", "deactivate"):
        child = subparsers.add_parser(command)
        child.add_argument("object_id")
    subparsers.add_parser("list")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        payload = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
        if payload.get("schema") != MULTI_STATE_SCHEMA:
            raise ValueError(f"unsupported object-state schema: {payload.get('schema')!r}")
        summary = {
            object_id: bool(record.get("active", True))
            for object_id, record in payload.get("objects", {}).items()
            if isinstance(record, dict)
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    set_object_active(args.state_json, args.object_id, args.command == "activate")


if __name__ == "__main__":
    main()
