"""Recompute a descriptive MLX benchmark summary from independent session files.

Example::

    python -m bench.summarize_mlx session-1.json session-2.json session-3.json \
        --output summary.json

Every arm's session median is recomputed from raw block timings. The reported
latency is the median of those session medians; ratios divide the aggregated
latencies. This is a descriptive summary, not a confidence interval. Matching
software, source hashes, device, geometry and methodology are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

AGGREGATION = (
    "Per-arm median of each session median; ratios divide these aggregated latencies. "
    "Descriptive, no confidence interval."
)
_GEOMETRY_KEYS = {
    "width",
    "batch",
    "block_size",
    "dcp_size",
    "dcp_rank",
    "dcp_interleave",
    "compact_valid_to_front",
}
_METADATA_KEYS = {
    "platform",
    "machine",
    "python",
    "versions",
    "device",
    "execution_backend",
    "source_sha256",
}
_SOURCE_FILES = {
    "silkern/contract.py",
    "silkern/mlx.py",
    "silkern/mlx_verify.py",
    "bench/bench_mlx.py",
}
_TIMING_METHOD = {
    "measurement": "synchronized end-to-end functional call",
    "units": "microseconds per call",
    "includes": ["Python dispatch", "output allocation", "execution", "synchronization"],
    "excludes": ["input generation", "oracle validation", "first-use compilation"],
    "order": "rotate arm order each block",
    "full_model_decode": False,
}


def _integer(value: Any, minimum: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _validate_session(data: Any) -> list[dict[str, float]]:
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] not in (1, 2)
    ):
        raise ValueError("unsupported or missing benchmark schema_version")
    if data.get("benchmark") != "silkern_mlx_localization":
        raise ValueError("input is not a SILKern MLX benchmark session")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or not _METADATA_KEYS <= metadata.keys():
        raise ValueError("session environment metadata is incomplete")
    if metadata["execution_backend"] != "apple_metal":
        raise ValueError("expected Apple Metal execution metadata")
    if any(
        not isinstance(metadata[key], str) or not metadata[key]
        for key in ("platform", "machine", "python")
    ):
        raise ValueError("missing or malformed environment identity")
    versions, device = metadata["versions"], metadata["device"]
    if (
        not isinstance(versions, dict)
        or any(not isinstance(versions.get(name), str) or not versions[name] for name in ("silkern", "mlx"))
        or not isinstance(device, dict)
        or not isinstance(device.get("device_name"), str)
        or not device["device_name"]
    ):
        raise ValueError("missing or malformed runtime/device metadata")
    hashes = metadata["source_sha256"]
    if (
        not isinstance(hashes, dict)
        or not _SOURCE_FILES <= hashes.keys()
        or any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in hashes.items()
        )
    ):
        raise ValueError("missing or malformed source hashes")
    method = data.get("methodology")
    if (
        not isinstance(method, dict)
        or any(method.get(key) != value for key, value in _TIMING_METHOD.items())
        or method.get("full_model_decode") is not False
    ):
        raise ValueError("unsupported or missing timing methodology")
    if any(not _integer(method.get(name)) for name in ("blocks", "iterations_per_block", "warmup")):
        raise ValueError("invalid timing block, iteration or warmup count")
    if type(method.get("seed")) is not int or not isinstance(method.get("include_compiled"), bool):
        raise ValueError("missing seed or compiled-mode methodology")
    if data["schema_version"] == 2:
        expected_checks = ["request_ids", "block_table", "token_indices"] if method["include_compiled"] else []
        if method.get("compiled_input_checks") != expected_checks:
            raise ValueError("missing or incomplete compiled-input correctness methodology")
    arms = (
        ("mlx", "metal", "mlx_compiled", "metal_compiled")
        if method["include_compiled"]
        else ("mlx", "metal")
    )
    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("session must contain benchmark cells")
    seen_geometry = set()
    medians = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("oracle_passed") is not True:
            raise ValueError(f"cell {index} did not pass oracle validation")
        if (
            data["schema_version"] == 2
            and cell.get("compiled_changing_inputs_passed") is not method["include_compiled"]
        ):
            raise ValueError(f"cell {index} has incomplete compiled-input correctness checks")
        geometry = cell.get("geometry")
        if not isinstance(geometry, dict) or set(geometry) != _GEOMETRY_KEYS:
            raise ValueError(f"cell {index} has malformed geometry")
        for name in _GEOMETRY_KEYS - {"compact_valid_to_front"}:
            if not _integer(geometry[name], 0 if name == "dcp_rank" else 1):
                raise ValueError(f"cell {index} has invalid {name}")
        if (
            not isinstance(geometry["compact_valid_to_front"], bool)
            or geometry["dcp_rank"] >= geometry["dcp_size"]
            or geometry["block_size"] % geometry["dcp_interleave"]
        ):
            raise ValueError(f"cell {index} has invalid localization geometry")
        geometry_key = json.dumps(geometry, sort_keys=True)
        if geometry_key in seen_geometry:
            raise ValueError(f"duplicate geometry in cell {index}")
        seen_geometry.add(geometry_key)
        if type(cell.get("seed")) is not int or cell["seed"] != method["seed"] + index:
            raise ValueError(f"cell {index} seed does not match benchmark methodology")
        fraction = cell.get("valid_fraction")
        if not _finite_number(fraction) or not 0 <= fraction <= 1:
            raise ValueError(f"cell {index} has invalid survivor fraction")
        results = cell.get("results")
        if not isinstance(results, dict) or set(results) != set(arms):
            raise ValueError(f"cell {index} has inconsistent benchmark arms")
        schedule = cell.get("schedule")
        if (
            not isinstance(schedule, list)
            or len(schedule) != method["blocks"]
            or any(
                order != list(arms[index % len(arms) :] + arms[: index % len(arms)])
                for index, order in enumerate(schedule)
            )
        ):
            raise ValueError(f"cell {index} has an invalid timing schedule")
        cell_medians = {}
        for arm in arms:
            result = results[arm]
            samples = result.get("block_samples_us") if isinstance(result, dict) else None
            if (
                not isinstance(samples, list)
                or len(samples) != method["blocks"]
                or any(not _finite_number(value) or value <= 0 for value in samples)
            ):
                raise ValueError(f"cell {index}/{arm} has invalid raw block timings")
            cell_medians[arm] = statistics.median(samples)
        medians.append(cell_medians)
    return medians


def summarize_sessions(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Summarize matching independent files, rejecting duplicates and mixtures."""
    files = [Path(path) for path in paths]
    if len(files) < 2:
        raise ValueError("at least two independent session files are required")
    if len({path.resolve() for path in files}) != len(files):
        raise ValueError("duplicate session file paths")
    sessions, all_medians = [], []
    content_hashes, timestamps = set(), set()
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            medians = _validate_session(data)
            timestamp = datetime.fromisoformat(data["created_at"])
            if timestamp.tzinfo is None:
                raise ValueError("session timestamp must include a time zone")
            # A copied measurement does not become independent when its file
            # name, timestamp, or cached summary fields are changed.
            raw_timings = [
                {name: result["block_samples_us"] for name, result in cell["results"].items()}
                for cell in data["cells"]
            ]
            digest = hashlib.sha256(json.dumps(raw_timings, sort_keys=True).encode()).hexdigest()
            if digest in content_hashes or timestamp in timestamps:
                raise ValueError("duplicate session raw timings or creation timestamp")
            content_hashes.add(digest)
            timestamps.add(timestamp)
            if sessions:
                reference = sessions[0]
                if data["schema_version"] != reference["schema_version"]:
                    raise ValueError("inconsistent benchmark schema versions")
                if data["metadata"] != reference["metadata"]:
                    raise ValueError("inconsistent source hashes or environment metadata")
                if data["methodology"] != reference["methodology"]:
                    raise ValueError("inconsistent timing methodology")
                if len(data["cells"]) != len(reference["cells"]):
                    raise ValueError("inconsistent geometry matrix")
                for cell, ref_cell in zip(data["cells"], reference["cells"], strict=True):
                    if any(
                        cell[key] != ref_cell[key]
                        for key in ("geometry", "seed", "valid_fraction", "schedule")
                    ):
                        raise ValueError("inconsistent geometry, input or timing schedule")
            sessions.append(data)
            all_medians.append(medians)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    summary = {
        "schema_version": 1,
        "aggregation": AGGREGATION,
        "sessions": [path.name for path in files],
        "session_count": len(files),
        "cells": [],
    }
    compiled = sessions[0]["methodology"]["include_compiled"]
    for index, first_cell in enumerate(sessions[0]["cells"]):
        results = {
            arm: statistics.median(session[index][arm] for session in all_medians)
            for arm in all_medians[0][index]
        }
        cell = {
            "geometry": first_cell["geometry"],
            "median_us": results,
            "eager_ratio": results["mlx"] / results["metal"],
        }
        if compiled:
            cell["compiled_ratio"] = results["mlx_compiled"] / results["metal_compiled"]
        summary["cells"].append(cell)
    for name in ("eager_ratio", "compiled_ratio") if compiled else ("eager_ratio",):
        summary[f"{name}_range"] = [
            min(cell[name] for cell in summary["cells"]),
            max(cell[name] for cell in summary["cells"]),
        ]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sessions", type=Path, nargs="+", help="independent benchmark session JSON files"
    )
    parser.add_argument("--output", type=Path, help="save summary JSON; otherwise print it")
    args = parser.parse_args(argv)
    try:
        if args.output and (
            args.output.resolve() in {path.resolve() for path in args.sessions}
            or (
                args.output.exists()
                and any(args.output.samefile(path) for path in args.sessions)
            )
        ):
            raise ValueError("output must not overwrite an input session")
        encoded = json.dumps(summarize_sessions(args.sessions), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except (OSError, ValueError) as exc:
        print(f"MLX summary: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
