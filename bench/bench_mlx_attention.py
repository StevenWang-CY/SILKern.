"""Measure a complete compiled selected-attention consumer on one Apple device.

The fixed default cells include localization, paged K/V gathers, masked softmax,
and recombination of two logical shards. One outer eval realizes each fresh
functional call. This is a single-head consumer diagnostic, not model decode or
distributed throughput. Run from a checkout; neither examples nor bench ship in
the library wheel. Use --summarize SESSION... to aggregate independent runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import struct
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from bench.bench_mlx import _measure, _summarize
from bench.summarize_mlx import _finite_number, _integer, _reject_constant
from silkern.contract import MAX_ROW_WIDTH
from silkern.errors import LocalizationError
from silkern.mlx_verify import _load_mlx, _metadata, _positive_int

DEFAULT_CELLS = ((1, 128), (8, 2048))  # (batch, selection width)
ARMS = ("mlx_compiled", "metal_compiled")
BLOCK_SIZE, SHARDS, KEY_DIM, VALUE_DIM, REQUESTS = 64, 2, 64, 64, 4
MAX_SELECTION_ELEMENTS = 65_536
MIN_SELECTION_WIDTH = 8
_SOURCE_FILES = {
    "silkern/contract.py",
    "silkern/mlx.py",
    "silkern/mlx_verify.py",
    "bench/bench_mlx.py",
    "bench/bench_mlx_attention.py",
    "examples/mlx_sparse_attention.py",
}
_PROTOCOL = {
    "measurement": "synchronized compiled selected-attention consumer",
    "units": "microseconds per call",
    "includes": [
        "localization",
        "paged K/V gather",
        "masked softmax",
        "logical-shard recombination",
        "Python dispatch",
        "output allocation",
        "synchronization",
    ],
    "excludes": ["fixture generation", "oracle validation", "first-use compilation"],
    "order": "rotate arm order each block",
    "full_model_decode": False,
    "distributed_execution": False,
    "oracle": "independent unsharded float64 attention and exact per-shard counts",
    "float_tolerance": {"relative": 1e-5, "absolute": 1e-6},
    "changing_input_checks": ["query", "request_ids", "token_indices"],
    "static_inputs": ["page tables", "key/value caches"],
}


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _fixture(batch, width, seed):
    """Build deterministic finite logical K/V, then page each shard over a NaN cache."""
    rng = random.Random(seed)
    table_width = max(2, (width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    local_limit = table_width * BLOCK_SIZE
    global_limit = local_limit * SHARDS
    logical_keys, logical_values = [], []
    for request in range(REQUESTS):
        logical_keys.append(
            [
                [
                    _float32(math.sin((token + 1) * (dimension + 1) * 0.013 + request * 0.3))
                    for dimension in range(KEY_DIM)
                ]
                for token in range(global_limit)
            ]
        )
        logical_values.append(
            [
                [
                    _float32(math.cos((token + 1) * (dimension + 1) * 0.009 + request * 0.7))
                    for dimension in range(VALUE_DIM)
                ]
                for token in range(global_limit)
            ]
        )
    requests = [(row * 3 + 1) % REQUESTS for row in range(batch)]
    query = [
        [
            _float32(math.sin((row + 1) * (dimension + 1) * 0.17) * 0.5)
            for dimension in range(KEY_DIM)
        ]
        for row in range(batch)
    ]
    rows = [
        [rng.randrange(global_limit) if rng.random() >= 0.1 else -1 for _ in range(width)]
        for _ in range(batch)
    ]
    special = [global_limit - 1, 0, 3, 3, -1, global_limit, 1, 2]
    for row in rows:
        row[: min(width, len(special))] = special[:width]
    if batch > 1:
        rows[-1] = [-1] * width
    shards = []
    for rank in range(SHARDS):
        pages = rng.sample(range(1, REQUESTS * table_width + 1), REQUESTS * table_width)
        table = [
            pages[request * table_width : (request + 1) * table_width]
            for request in range(REQUESTS)
        ]
        # Every slot no owned token occupies, including all of placeholder page 0,
        # holds NaN: a consumer that gathers a padded slot without masking the
        # gathered values turns the output NaN and fails the qualification gate.
        keys = [[math.nan] * KEY_DIM for _ in range((len(pages) + 1) * BLOCK_SIZE)]
        values = [[math.nan] * VALUE_DIM for _ in range((len(pages) + 1) * BLOCK_SIZE)]
        for request in range(REQUESTS):
            for local in range(local_limit):
                slot = table[request][local // BLOCK_SIZE] * BLOCK_SIZE + local % BLOCK_SIZE
                token = local * SHARDS + rank
                keys[slot], values[slot] = (
                    logical_keys[request][token],
                    logical_values[request][token],
                )
        shards.append((table, keys, values))
    return query, requests, rows, shards, logical_keys, logical_values


def _reference(query, requests, rows, logical_keys, logical_values):
    """Float64 scalar attention in global selection order; no localization oracle."""
    output = []
    counts = [[] for _ in range(SHARDS)]
    for q, request, row in zip(query, requests, rows, strict=True):
        valid_request = 0 <= request < len(logical_keys)
        selected = [
            token for token in row if valid_request and 0 <= token < len(logical_keys[request])
        ]
        for rank in range(SHARDS):
            counts[rank].append(sum(token % SHARDS == rank for token in selected))
        if not selected:
            output.append([0.0] * VALUE_DIM)
            continue
        logits = [
            math.fsum(a * b for a, b in zip(q, logical_keys[request][token], strict=True))
            / math.sqrt(KEY_DIM)
            for token in selected
        ]
        maximum = max(logits)
        weights = [math.exp(logit - maximum) for logit in logits]
        denominator = math.fsum(weights)
        output.append(
            [
                math.fsum(
                    weight * logical_values[request][token][dimension]
                    for weight, token in zip(weights, selected, strict=True)
                )
                / denominator
                for dimension in range(VALUE_DIM)
            ]
        )
    return output, counts


def _check_output(name, observed, counts, expected, expected_counts):
    if counts != expected_counts or len(observed) != len(expected):
        raise LocalizationError(f"{name} consumer oracle mismatch: shape or per-shard counts")
    error = 0.0
    for actual, wanted in zip(observed, expected, strict=True):
        if len(actual) != len(wanted):
            raise LocalizationError(f"{name} consumer oracle mismatch: output dimensions")
        for value, reference in zip(actual, wanted, strict=True):
            if not math.isfinite(value) or not math.isclose(
                value, reference, rel_tol=1e-5, abs_tol=1e-6
            ):
                raise LocalizationError(f"{name} consumer oracle mismatch: attention values")
            error = max(error, abs(value - reference))
    return error


def _qualify_inputs(mx, functions, arrays, query, requests, rows, logical_keys, logical_values):
    """Check original inputs and independent mutations; page/cache fixtures stay fixed."""
    cases = (
        ("original", query, requests, rows),
        ("query", [[-value for value in row] for row in query], requests, rows),
        ("request_ids", query, [(request + 1) % REQUESTS for request in requests], rows),
        ("token_indices", query, requests, [[-1] * len(row) for row in rows]),
    )
    errors = {}
    for changed, q, req, tokens in cases:
        expected, expected_counts = _reference(q, req, tokens, logical_keys, logical_values)
        args = (
            mx.array(q, dtype=mx.float32),
            mx.array(req, dtype=mx.int32),
            mx.array(tokens, dtype=mx.int32),
            arrays[3],
        )
        mx.eval(args)
        for name, function in functions.items():
            out, counts = function(*args)
            mx.eval(out, counts)
            error = _check_output(
                f"{name}/{changed}",
                out.tolist(),
                [c.tolist() for c in counts],
                expected,
                expected_counts,
            )
            errors[name] = max(errors.get(name, 0.0), error)
    return errors


def benchmark_attention(*, cells=DEFAULT_CELLS, warmup=10, iterations=50, blocks=12, seed=11):
    """Qualify and measure bounded compiled consumers on a Metal GPU."""
    normalized = []
    for cell in cells:
        if not isinstance(cell, (tuple, list)) or len(cell) != 2:
            raise LocalizationError("each cell must be (batch, width)")
        batch, width = (
            _positive_int(name, value) for name, value in zip(("batch", "width"), cell, strict=True)
        )
        if (
            not MIN_SELECTION_WIDTH <= width <= MAX_ROW_WIDTH
            or batch * width > MAX_SELECTION_ELEMENTS
        ):
            raise LocalizationError(
                "consumer fixture exceeds the supported width or selection budget"
            )
        normalized.append((batch, width))
    if not normalized or len(set(normalized)) != len(normalized):
        raise LocalizationError("cells must contain distinct nonempty geometries")
    warmup, iterations, blocks = (
        _positive_int(name, value)
        for name, value in (("warmup", warmup), ("iterations", iterations), ("blocks", blocks))
    )
    if type(seed) is not int:
        raise LocalizationError("seed must be an integer")
    mx, reason = _load_mlx()
    if reason:
        raise LocalizationError(reason)
    from examples.mlx_sparse_attention import selected_attention

    stream = mx.default_stream(mx.gpu)
    metadata = _metadata(mx)
    root = Path(__file__).resolve().parents[1]
    metadata["source_sha256"].update(
        {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _SOURCE_FILES}
    )
    report = {
        "schema_version": 1,
        "benchmark": "silkern_mlx_selected_attention",
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "methodology": {
            **_PROTOCOL,
            "warmup": warmup,
            "iterations_per_block": iterations,
            "blocks": blocks,
            "seed": seed,
        },
        "cells": [],
    }
    for index, (batch, width) in enumerate(normalized):
        case_seed = seed + index
        query, requests, rows, shards, logical_keys, logical_values = _fixture(
            batch, width, case_seed
        )
        arrays = (
            mx.array(query, dtype=mx.float32),
            mx.array(requests, dtype=mx.int32),
            mx.array(rows, dtype=mx.int32),
            [
                (
                    mx.array(table, dtype=mx.int32),
                    mx.array(keys, dtype=mx.float32),
                    mx.array(values, dtype=mx.float32),
                )
                for table, keys, values in shards
            ],
        )
        mx.eval(arrays)
        functions = {
            f"{backend}_compiled": mx.compile(
                partial(selected_attention, block_size=BLOCK_SIZE, backend=backend, stream=stream)
            )
            for backend in ("mlx", "metal")
        }
        _, expected_counts = _reference(query, requests, rows, logical_keys, logical_values)
        errors = _qualify_inputs(
            mx, functions, arrays, query, requests, rows, logical_keys, logical_values
        )
        arms = {name: partial(function, *arrays) for name, function in functions.items()}
        samples, schedule = _measure(
            mx, arms, warmup=warmup, iterations=iterations, blocks=blocks, stream=stream
        )
        results = {name: _summarize(values) for name, values in samples.items()}
        report["cells"].append(
            {
                "geometry": {
                    "batch": batch,
                    "width": width,
                    "block_size": BLOCK_SIZE,
                    "dcp_size": SHARDS,
                    "dcp_interleave": 1,
                    "key_dim": KEY_DIM,
                    "value_dim": VALUE_DIM,
                    "request_count": REQUESTS,
                    "table_width": len(shards[0][0][0]),
                },
                "seed": case_seed,
                "oracle_passed": True,
                "changing_inputs_passed": True,
                "oracle_max_abs_error": errors,
                "valid_counts": expected_counts,
                "schedule": schedule,
                "results": results,
                "compiled_ratio": results["mlx_compiled"]["median_us"]
                / results["metal_compiled"]["median_us"],
            }
        )
    return report


def summarize_attention(paths):
    """Aggregate per-arm session medians without treating blocks as independent runs."""
    paths = [Path(path) for path in paths]
    if len(paths) < 2 or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("at least two distinct session files are required")
    sessions, medians, timestamps, raw_hashes = [], [], set(), set()
    for path in paths:
        data = json.loads(path.read_text(), parse_constant=_reject_constant)
        try:
            if (
                type(data["schema_version"]) is not int
                or data["schema_version"] != 1
                or data["benchmark"] != "silkern_mlx_selected_attention"
            ):
                raise ValueError("expected a selected-attention benchmark session")
            timestamp = datetime.fromisoformat(data["created_at"])
            if timestamp.tzinfo is None or timestamp in timestamps:
                raise ValueError("missing time zone or duplicate timestamp")
            timestamps.add(timestamp)
            metadata, method = data["metadata"], data["methodology"]
            hashes = metadata["source_sha256"]
            if (
                metadata.get("execution_backend") != "apple_metal"
                or any(
                    not isinstance(metadata.get(key), str) or not metadata[key]
                    for key in ("platform", "machine", "python")
                )
                or any(
                    not isinstance(metadata["versions"].get(key), str)
                    or not metadata["versions"][key]
                    for key in ("silkern", "mlx")
                )
                or not isinstance(metadata["device"].get("device_name"), str)
                or not metadata["device"]["device_name"]
                or not _SOURCE_FILES <= hashes.keys()
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                    for value in hashes.values()
                )
            ):
                raise ValueError("incomplete source or device provenance")
            if (
                any(method.get(key) != value for key, value in _PROTOCOL.items())
                or method.get("full_model_decode") is not False
                or method.get("distributed_execution") is not False
                or any(
                    not _integer(method.get(key))
                    for key in ("warmup", "iterations_per_block", "blocks")
                )
                or type(method.get("seed")) is not int
            ):
                raise ValueError("invalid consumer timing methodology")
            if not isinstance(data["cells"], list) or not data["cells"]:
                raise ValueError("missing consumer cells")
            session_medians, seen = [], set()
            for index, cell in enumerate(data["cells"]):
                geometry = cell["geometry"]
                geometry_key = json.dumps(geometry, sort_keys=True)
                if (
                    geometry_key in seen
                    or cell.get("oracle_passed") is not True
                    or cell.get("changing_inputs_passed") is not True
                ):
                    raise ValueError("duplicate or unqualified consumer cell")
                seen.add(geometry_key)
                if (
                    set(geometry)
                    != {
                        "batch",
                        "width",
                        "block_size",
                        "dcp_size",
                        "dcp_interleave",
                        "key_dim",
                        "value_dim",
                        "request_count",
                        "table_width",
                    }
                    or any(not _integer(value) for value in geometry.values())
                    or not MIN_SELECTION_WIDTH <= geometry["width"] <= MAX_ROW_WIDTH
                    or geometry["batch"] * geometry["width"] > MAX_SELECTION_ELEMENTS
                    or any(
                        geometry[key] != value
                        for key, value in (
                            ("block_size", BLOCK_SIZE),
                            ("dcp_size", SHARDS),
                            ("dcp_interleave", 1),
                            ("key_dim", KEY_DIM),
                            ("value_dim", VALUE_DIM),
                            ("request_count", REQUESTS),
                        )
                    )
                    or geometry["table_width"]
                    != max(2, (geometry["width"] + BLOCK_SIZE - 1) // BLOCK_SIZE)
                    or type(cell.get("seed")) is not int
                    or cell["seed"] != method["seed"] + index
                ):
                    raise ValueError("invalid consumer geometry or seed")
                schedule = cell["schedule"]
                if (
                    not isinstance(schedule, list)
                    or len(schedule) != method["blocks"]
                    or any(
                        order != list(ARMS[block % 2 :] + ARMS[: block % 2])
                        for block, order in enumerate(schedule)
                    )
                ):
                    raise ValueError("invalid rotating schedule")
                counts, errors = cell["valid_counts"], cell["oracle_max_abs_error"]
                if (
                    not isinstance(counts, list)
                    or len(counts) != SHARDS
                    or any(
                        not isinstance(row, list)
                        or len(row) != geometry["batch"]
                        or any(not _integer(count, 0) or count > geometry["width"] for count in row)
                        for row in counts
                    )
                    or any(sum(row) > geometry["width"] for row in zip(*counts, strict=True))
                ):
                    raise ValueError("invalid per-shard survivor counts")
                if (
                    not isinstance(errors, dict)
                    or set(errors) != set(ARMS)
                    or any(not _finite_number(error) or error < 0 for error in errors.values())
                ):
                    raise ValueError("invalid oracle error measurements")
                if set(cell["results"]) != set(ARMS):
                    raise ValueError("inconsistent consumer arms")
                arm_medians = {}
                for arm in ARMS:
                    values = cell["results"][arm]["block_samples_us"]
                    if (
                        not isinstance(values, list)
                        or len(values) != method["blocks"]
                        or any(not _finite_number(value) or value <= 0 for value in values)
                    ):
                        raise ValueError("invalid raw consumer timings")
                    arm_medians[arm] = statistics.median(values)
                session_medians.append(arm_medians)
            raw = [
                {arm: cell["results"][arm]["block_samples_us"] for arm in ARMS}
                for cell in data["cells"]
            ]
            digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            if digest in raw_hashes:
                raise ValueError("duplicate raw consumer timings")
            raw_hashes.add(digest)
            if sessions:
                first = sessions[0]
                if (
                    metadata != first["metadata"]
                    or method != first["methodology"]
                    or len(data["cells"]) != len(first["cells"])
                    or any(
                        (cell["geometry"], cell["seed"], cell["valid_counts"])
                        != (reference["geometry"], reference["seed"], reference["valid_counts"])
                        for cell, reference in zip(data["cells"], first["cells"], strict=True)
                    )
                ):
                    raise ValueError("inconsistent consumer sessions")
            sessions.append(data)
            medians.append(session_medians)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path}: malformed consumer record") from exc
    summary = {
        "schema_version": 1,
        "benchmark": "silkern_mlx_selected_attention_summary",
        "aggregation": "Per-arm median of session medians; descriptive ratios, no confidence interval.",
        "sessions": [path.name for path in paths],
        "session_count": len(paths),
        "metadata": sessions[0]["metadata"],
        "methodology": sessions[0]["methodology"],
        "cells": [],
    }
    for index, cell in enumerate(sessions[0]["cells"]):
        result = {
            arm: statistics.median(session[index][arm] for session in medians) for arm in ARMS
        }
        summary["cells"].append(
            {
                "geometry": cell["geometry"],
                "median_us": result,
                "compiled_ratio": result["mlx_compiled"] / result["metal_compiled"],
            }
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure a complete compiled selected-attention consumer on one Apple device.")
    parser.add_argument(
        "--summarize",
        type=Path,
        nargs="+",
        help="aggregate existing session JSON files without a device",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, help="save JSON; otherwise print it")
    args = parser.parse_args(argv)
    try:
        if (
            args.summarize
            and args.output
            and any(
                args.output.resolve() == path.resolve()
                or (args.output.exists() and args.output.samefile(path))
                for path in args.summarize
            )
        ):
            raise ValueError("output must not overwrite an input session")
        report = (
            summarize_attention(args.summarize)
            if args.summarize
            else benchmark_attention(
                warmup=args.warmup, iterations=args.iterations, blocks=args.blocks, seed=args.seed
            )
        )
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        else:
            print(encoded, end="")
    except (LocalizationError, OSError, RuntimeError, ValueError) as exc:
        print(f"MLX attention benchmark: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
