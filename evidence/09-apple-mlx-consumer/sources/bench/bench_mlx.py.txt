"""Benchmark exact sparse-index localization on Apple silicon with MLX.

Each geometry is checked against the Python oracle before measurement. Every
timed call creates and evaluates fresh outputs; results include Python dispatch,
allocation and synchronization. Warmup excludes first-use compilation. Arm order
rotates between blocks, and the JSON retains raw block timings and environment
metadata. These are local converter measurements, not full-model decode results.
Compiled arms must also consume independently changed request, table, and token
arrays correctly before timing starts.

Example: ``python -m bench.bench_mlx --width 128 2048 4096 --batch 1 8 --output mlx.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import Any

from silkern.contract import MAX_ROW_WIDTH, _validate_dcp_config, localize_reference
from silkern.errors import LocalizationError
from silkern.mlx_verify import _load_mlx, _metadata, _positive_int, _validate_fixture

COMPILED_INPUT_CHECKS = ("request_ids", "block_table", "token_indices")


def _qualify_compiled_inputs(mx, functions, *, width, batch, table_width, common):
    """Reject captured array values in the actual functions that will be timed.

    The canonical fixture has the measured shapes and at least one valid token
    per row even when the random timing fixture has no survivors. Each input is
    changed independently, so a compiled function must consume all three arrays.
    """
    req_ids = [0] * batch
    table = [[request * table_width + page for page in range(table_width)] for request in range(4)]
    rows = [[common["dcp_rank"] * common["dcp_interleave"]] * width for _ in range(batch)]
    cases = (
        ("baseline", req_ids, table, rows),
        ("request_ids", [1] * batch, table, rows),
        ("block_table", req_ids, [[page + 1 for page in row] for row in table], rows),
        ("token_indices", req_ids, table, [[-1] * width for _ in range(batch)]),
    )
    for changed, requests, pages, tokens in cases:
        expected, expected_counts = localize_reference(requests, pages, tokens, **common)
        arrays = tuple(mx.array(data, dtype=mx.int32) for data in (requests, pages, tokens))
        mx.eval(*arrays)
        for name, function in functions.items():
            out, counts = function(*arrays)
            mx.eval(out, counts)
            if out.tolist() != expected or counts.tolist() != expected_counts:
                raise LocalizationError(
                    f"{name} changing-input oracle mismatch for {changed} "
                    f"at width={width}, batch={batch}"
                )


def _case(width: int, batch: int, block_size: int, dcp_size: int, seed: int):
    rng = random.Random(seed)
    table_width = max(17, math.ceil(width / block_size))
    req_ids = [(row * 3 + 1) % 4 for row in range(batch)]
    table = [rng.sample(range(max(4096, 2 * table_width)), table_width) for _ in range(4)]
    limit = block_size * table_width * dcp_size
    rows = [
        [rng.randrange(limit) if rng.random() >= 0.1 else -1 for _ in range(width)]
        for _ in range(batch)
    ]
    return req_ids, table, rows


def _time_block(mx, launch: Callable, *, iterations: int, stream) -> float:
    """End-to-end microseconds per call; never time repeated eval of one graph."""
    mx.synchronize(stream)
    start = time.perf_counter_ns()
    for _ in range(iterations):
        mx.eval(launch())
    elapsed = time.perf_counter_ns() - start
    return elapsed / (iterations * 1000.0)


def _measure(mx, arms: dict[str, Callable], *, warmup: int, iterations: int, blocks: int, stream):
    for _ in range(warmup):
        for launch in arms.values():
            mx.eval(launch())
    names = list(arms)
    samples = {name: [] for name in names}
    schedule = []
    for block in range(blocks):
        offset = block % len(names)
        order = names[offset:] + names[:offset]
        schedule.append(order)
        for name in order:
            samples[name].append(_time_block(mx, arms[name], iterations=iterations, stream=stream))
    return samples, schedule


def _summarize(samples: Sequence[float]) -> dict[str, Any]:
    return {
        "block_samples_us": list(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "stdev_us": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def benchmark_mlx(
    *,
    widths: Sequence[int] = (128, 512, 2048, 4096),
    batches: Sequence[int] = (1, 8, 32),
    block_size: int = 64,
    dcp_size: int = 2,
    dcp_rank: int = 0,
    dcp_interleave: int = 1,
    compact_valid_to_front: bool = True,
    warmup: int = 10,
    iterations: int = 50,
    blocks: int = 12,
    seed: int = 11,
    include_compiled: bool = False,
) -> dict[str, Any]:
    """Qualify and compare native MLX and custom Metal for each geometry.

    The measurement unit is one synchronized functional call, including output
    allocation. Samples within a process are descriptive repeated measurements,
    not independent experimental sessions or a confidence interval.
    """
    widths, batches = list(widths), list(batches)
    if not widths or not batches:
        raise LocalizationError("widths and batches must both be nonempty")
    for width in widths:
        if _positive_int("width", width) > MAX_ROW_WIDTH:
            raise LocalizationError(f"width must not exceed {MAX_ROW_WIDTH}")
    for batch in batches:
        _positive_int("batch", batch)
    for name, value in (
        ("block_size", block_size),
        ("warmup", warmup),
        ("iterations", iterations),
        ("blocks", blocks),
    ):
        _positive_int(name, value)
    _validate_dcp_config(dcp_size, dcp_rank, dcp_interleave)
    block_size, dcp_size, dcp_rank, dcp_interleave = map(
        int, (block_size, dcp_size, dcp_rank, dcp_interleave)
    )
    if block_size % dcp_interleave:
        raise LocalizationError("block_size must be divisible by dcp_interleave")
    if not isinstance(compact_valid_to_front, bool):
        raise LocalizationError("compact_valid_to_front must be a bool")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise LocalizationError("seed must be an integer")
    if not isinstance(include_compiled, bool):
        raise LocalizationError("include_compiled must be a bool")
    widths, batches = list(map(int, widths)), list(map(int, batches))
    if len(set(widths)) != len(widths) or len(set(batches)) != len(batches):
        raise LocalizationError("widths and batches must not contain duplicate geometries")
    warmup, iterations, blocks = map(int, (warmup, iterations, blocks))
    for width in widths:
        for batch in batches:
            _validate_fixture(
                width, batch, block_size, dcp_size, max(17, math.ceil(width / block_size))
            )
    mx, reason = _load_mlx()
    if reason:
        raise LocalizationError(reason)
    from silkern.mlx import localize_mlx

    stream = mx.default_stream(mx.gpu)
    metadata = _metadata(mx)
    metadata.setdefault("source_sha256", {})["bench/bench_mlx.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    common = dict(
        block_size=block_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_interleave=dcp_interleave,
        compact_valid_to_front=compact_valid_to_front,
    )
    report = {
        "schema_version": 2,
        "benchmark": "silkern_mlx_localization",
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "methodology": {
            "measurement": "synchronized end-to-end functional call",
            "units": "microseconds per call",
            "includes": ["Python dispatch", "output allocation", "execution", "synchronization"],
            "excludes": ["input generation", "oracle validation", "first-use compilation"],
            "order": "rotate arm order each block",
            "warmup": warmup,
            "iterations_per_block": iterations,
            "blocks": blocks,
            "seed": int(seed),
            "full_model_decode": False,
            "include_compiled": include_compiled,
            "compiled_input_checks": list(COMPILED_INPUT_CHECKS) if include_compiled else [],
        },
        "cells": [],
    }
    for width in widths:
        for batch in batches:
            case_seed = int(seed) + len(report["cells"])
            req_ids, table_data, rows = _case(width, batch, block_size, dcp_size, case_seed)
            expected, expected_counts = localize_reference(req_ids, table_data, rows, **common)
            req = mx.array(req_ids, dtype=mx.int32)
            table = mx.array(table_data, dtype=mx.int32)
            tokens = mx.array(rows, dtype=mx.int32)
            mx.eval(req, table, tokens)
            arms = {
                backend: partial(
                    localize_mlx, req, table, tokens, **common, backend=backend, stream=stream
                )
                for backend in ("mlx", "metal")
            }
            compiled_functions = {}
            if include_compiled:
                # Arrays are dynamic function arguments: compiling a zero-argument
                # closure could cache a constant result and benchmark no work.
                for backend in ("mlx", "metal"):
                    compiled = mx.compile(
                        partial(localize_mlx, **common, backend=backend, stream=stream)
                    )
                    name = f"{backend}_compiled"
                    compiled_functions[name] = compiled
                    arms[name] = partial(compiled, req, table, tokens)
            # Fail before timing if either implementation violates the oracle.
            for name, launch in arms.items():
                out, counts = launch()
                mx.eval(out, counts)
                if out.tolist() != expected or counts.tolist() != expected_counts:
                    raise LocalizationError(
                        f"{name} oracle mismatch at width={width}, batch={batch}"
                    )
            if compiled_functions:
                _qualify_compiled_inputs(
                    mx, compiled_functions, width=width, batch=batch,
                    table_width=len(table_data[0]), common=common,
                )
            samples, schedule = _measure(
                mx, arms, warmup=warmup, iterations=iterations, blocks=blocks, stream=stream
            )
            results = {name: _summarize(values) for name, values in samples.items()}
            cell = {
                "geometry": {"width": width, "batch": batch, **common},
                "seed": case_seed,
                "oracle_passed": True,
                "compiled_changing_inputs_passed": include_compiled,
                "valid_fraction": sum(expected_counts) / (width * batch),
                "schedule": schedule,
                "results": results,
                "mlx_over_metal": results["mlx"]["median_us"] / results["metal"]["median_us"],
            }
            if include_compiled:
                cell["mlx_compiled_over_metal_compiled"] = (
                    results["mlx_compiled"]["median_us"] / results["metal_compiled"]["median_us"]
                )
                cell["mlx_compiled_over_metal"] = (
                    results["mlx_compiled"]["median_us"] / results["metal"]["median_us"]
                )
            report["cells"].append(cell)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, nargs="+", default=[128, 512, 2048, 4096])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--dcp-rank", type=int, default=0)
    parser.add_argument("--dcp-interleave", type=int, default=1)
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", "--replays", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--compiled", action="store_true", help="also compare both backends under mx.compile"
    )
    parser.add_argument("--output", type=Path, help="save metadata and raw timings as JSON")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the summary")
    args = parser.parse_args(argv)
    try:
        report = benchmark_mlx(
            widths=args.width,
            batches=args.batch,
            block_size=args.block_size,
            dcp_size=args.dcp_size,
            dcp_rank=args.dcp_rank,
            dcp_interleave=args.dcp_interleave,
            compact_valid_to_front=not args.no_compact,
            warmup=args.warmup,
            iterations=args.iterations,
            blocks=args.blocks,
            seed=args.seed,
            include_compiled=args.compiled,
        )
    except (LocalizationError, RuntimeError) as exc:
        print(f"MLX benchmark: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    if args.json:
        print(encoded, end="")
    else:
        device = report["metadata"]["device"]
        print(f"Apple device: {device.get('device_name', device)}")
        print("Synchronized end-to-end latency; includes dispatch, allocation and evaluation.")
        print(" width  batch      MLX us    Metal us   MLX/Metal")
        for cell in report["cells"]:
            geometry, results = cell["geometry"], cell["results"]
            print(
                f"{geometry['width']:6d} {geometry['batch']:6d} "
                f"{results['mlx']['median_us']:11.2f} {results['metal']['median_us']:11.2f} "
                f"{cell['mlx_over_metal']:10.2f}x"
            )
        if args.output:
            print(f"Raw results: {args.output}")
        if args.compiled:
            print("\nBoth backends under mx.compile with dynamic input arrays:")
            print(" width  batch      MLX us    Metal us   MLX/Metal")
            for cell in report["cells"]:
                geometry, results = cell["geometry"], cell["results"]
                print(
                    f"{geometry['width']:6d} {geometry['batch']:6d} "
                    f"{results['mlx_compiled']['median_us']:11.2f} "
                    f"{results['metal_compiled']['median_us']:11.2f} "
                    f"{cell['mlx_compiled_over_metal_compiled']:10.2f}x"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
