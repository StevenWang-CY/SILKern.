"""Validate Apple MLX localization against the independent Python contract.

This module has no import-time MLX dependency and never selects a CUDA device.
Run ``python -m silkern.mlx_verify --require-device --json`` on Apple silicon.
The report covers functional results, stable order, repeatability, input
immutability and safe handling of out-of-range request ids. It does not claim
CUDA graph capture or allocation-free execution for the functional MLX API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

from silkern.contract import MAX_ROW_WIDTH, _validate_dcp_config, localize_reference
from silkern.errors import LocalizationError

MLX_CHECKS = ("oracle", "order", "determinism", "immutability", "shape", "dtype", "request_bounds")
MLX_BACKENDS = ("mlx", "metal")
MAX_FIXTURE_ELEMENTS = 4_194_304
_INT32_MAX = 2**31 - 1
DEFAULT_MLX_MATRIX: tuple[dict[str, int | bool], ...] = tuple(
    dict(
        width=width,
        block_size=block_size,
        dcp_size=size,
        dcp_rank=rank,
        dcp_interleave=interleave,
        compact_valid_to_front=compact,
    )
    for width, block_size, size, rank, interleave in (
        (1, 64, 1, 0, 1),
        (7, 32, 2, 1, 2),
        (31, 64, 4, 3, 4),
        (32, 64, 2, 0, 1),
        (33, 32, 8, 7, 1),
        (127, 64, 4, 1, 2),
        (128, 64, 2, 1, 1),
        (129, 64, 8, 7, 4),
        (513, 32, 4, 3, 4),
        (1024, 64, 1, 0, 2),
        (2049, 64, 2, 0, 1),
        (4096, 64, 8, 7, 2),
    )
    for compact in (False, True)
)


@dataclass
class MLXCellReport:
    """A cell passes only after every advertised check has run successfully."""

    backend: str
    geometry: dict[str, int | bool]
    checks: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and set(self.checks) == set(MLX_CHECKS)
            and all(self.checks[name] is True for name in MLX_CHECKS)
        )

    def failures(self) -> list[str]:
        if self.error is not None:
            return [self.error]
        return [name for name in MLX_CHECKS if self.checks.get(name) is not True] + [
            f"unexpected check: {name}" for name in self.checks if name not in MLX_CHECKS
        ]


@dataclass
class MLXConformanceReport:
    cells: list[MLXCellReport] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped is None and bool(self.cells) and all(cell.ok for cell in self.cells)

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ok": self.ok,
            "skipped": self.skipped,
            "metadata": self.metadata,
            "cells": [{**asdict(cell), "ok": cell.ok} for cell in self.cells],
        }

    def summary(self) -> str:
        if self.skipped is not None:
            return f"MLX conformance skipped: {self.skipped}"
        passed = sum(cell.ok for cell in self.cells)
        lines = [f"SILKern MLX conformance: {passed}/{len(self.cells)} cells passed"]
        for cell in self.cells:
            if not cell.ok:
                geometry = " ".join(f"{key}={value}" for key, value in cell.geometry.items())
                lines.append(f"  FAIL {cell.backend}: {geometry}: {', '.join(cell.failures())}")
        return "\n".join(lines)


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise LocalizationError(f"{name} must be a positive integer")
    return int(value)


def _load_mlx() -> tuple[Any, str | None]:
    """Find Metal explicitly; a Linux MLX CUDA installation is never exercised."""
    try:
        import mlx.core as mx
    except (ImportError, OSError) as exc:
        return None, f"MLX is unavailable; install silkern[mlx] on Apple silicon ({exc})"
    try:
        if not mx.metal.is_available():
            return None, "no Apple Metal device is available"
    except RuntimeError as exc:
        return None, f"Apple Metal initialization failed: {exc}"
    return mx, None


def _validate_fixture(
    width: int, batch: int, block_size: int, dcp_size: int, table_width: int
) -> None:
    """Bound host fixtures and ensure their generated indices fit signed int32."""
    if batch * width > MAX_FIXTURE_ELEMENTS:
        raise LocalizationError(
            f"diagnostic fixtures must not exceed {MAX_FIXTURE_ELEMENTS} elements"
        )
    if block_size * 4096 > _INT32_MAX or block_size * (table_width * dcp_size + 1) > _INT32_MAX:
        raise LocalizationError("geometry exceeds the signed int32 range of diagnostic fixtures")


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        f"silkern/{name}": hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in ("contract.py", "mlx.py", "mlx_verify.py")
    }


def _metadata(mx: Any) -> dict[str, Any]:
    versions = {}
    for package in ("silkern", "mlx"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
        "device": mx.device_info(mx.gpu),
        "execution_backend": "apple_metal",
        "source_sha256": _source_hashes(),
    }


def _random_case(
    *,
    width: int,
    batch: int,
    block_size: int,
    dcp_size: int,
    seed: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Generate unsorted duplicate selections, sentinels and fragmented pages."""
    rng = random.Random(seed)
    requests, table_width = 4, 17
    req_ids = [(row * 3 + 1) % requests for row in range(batch)]
    table = [[rng.randrange(4096) for _ in range(table_width)] for _ in range(requests)]
    for row in table:
        row[0], row[1], row[-1] = -1, -2, 0
    limit = block_size * table_width * dcp_size
    rows = []
    # Including the first whole ownership period ensures last-rank coverage.
    special = [-1, limit, limit - 1, 0, 0, -9, *range(min(width, block_size * dcp_size))]
    for row_id in range(batch):
        values = [rng.randrange(-block_size, limit + block_size) for _ in range(width)]
        prefix = special if row_id % 2 == 0 else list(reversed(special))
        values[: min(width, len(prefix))] = prefix[:width]
        if row_id == batch - 1 and batch > 1:
            values = [-1] * width  # An entirely empty survivor prefix.
        rows.append(values)
    return req_ids, table, rows


def _expected_order(
    row: Sequence[int],
    table_row: Sequence[int],
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
) -> list[int]:
    """Independent order derivation; page values do not determine validity."""
    owned = [
        token for token in row if token >= 0 and (token // dcp_interleave) % dcp_size == dcp_rank
    ]
    local = [
        (token // (dcp_size * dcp_interleave)) * dcp_interleave + token % dcp_interleave
        for token in owned
    ]
    return [
        table_row[token // block_size] * block_size + token % block_size
        for token in local
        if token // block_size < len(table_row)
    ]


def _run_cell(mx, backend, geometry, *, batch, seed, repeats) -> MLXCellReport:
    from silkern.mlx import localize_mlx

    cell = MLXCellReport(backend=backend, geometry=dict(geometry))
    try:
        width = geometry["width"]
        common = {key: value for key, value in geometry.items() if key != "width"}
        req_ids, table_data, rows = _random_case(
            width=width,
            batch=batch,
            block_size=geometry["block_size"],
            dcp_size=geometry["dcp_size"],
            seed=seed,
        )
        expected, expected_counts = localize_reference(req_ids, table_data, rows, **common)
        # Exercise non-contiguous arrays without changing the logical inputs.
        req = mx.array([value for value in req_ids for _ in range(2)], dtype=mx.int32)[::2]
        table = mx.array(
            [[value for value in row for _ in range(2)] for row in table_data], dtype=mx.int32
        )[:, ::2]
        tokens = mx.array(
            [[value for value in row for _ in range(2)] for row in rows], dtype=mx.int32
        )[:, ::2]
        mx.eval(req, table, tokens)

        def launch(requests=req):
            return localize_mlx(
                requests, table, tokens, **common, backend=backend, stream=mx.default_stream(mx.gpu)
            )

        out, counts = launch()
        mx.eval(out, counts)
        actual, actual_counts = out.tolist(), counts.tolist()
        cell.checks["oracle"] = actual == expected and actual_counts == expected_counts
        cell.checks["shape"] = out.shape == (batch, width) and counts.shape == (batch,)
        cell.checks["dtype"] = out.dtype == mx.int32 and counts.dtype == mx.int32
        compact = common["compact_valid_to_front"] and common["dcp_size"] > 1
        order_common = {
            key: value for key, value in common.items() if key != "compact_valid_to_front"
        }
        cell.checks["order"] = all(
            (
                actual[index][: actual_counts[index]]
                == _expected_order(row, table_data[req_ids[index]], **order_common)
                and actual[index][actual_counts[index] :] == [-1] * (width - actual_counts[index])
            )
            if compact
            else actual[index] == expected[index]
            for index, row in enumerate(rows)
        )
        cell.checks["determinism"] = True
        for _ in range(repeats):
            again, again_counts = launch()
            mx.eval(again, again_counts)
            cell.checks["determinism"] &= (
                again.tolist() == actual and again_counts.tolist() == actual_counts
            )
        cell.checks["immutability"] = (
            req.tolist() == req_ids and table.tolist() == table_data and tokens.tolist() == rows
        )
        invalid_req = mx.array(
            [-1 if index % 2 else len(table_data) for index in range(batch)], dtype=mx.int32
        )
        invalid_out, invalid_counts = launch(invalid_req)
        mx.eval(invalid_out, invalid_counts)
        cell.checks["request_bounds"] = (
            invalid_out.tolist() == [[-1] * width for _ in range(batch)]
            and invalid_counts.tolist() == [0] * batch
        )
    except Exception as exc:  # Report the exact failing geometry, including lazy runtime errors.
        cell.error = f"{type(exc).__name__}: {exc}"
    return cell


def conformance_mlx(
    *,
    backend: str = "both",
    matrix: Sequence[dict[str, int | bool]] | None = None,
    batch: int = 4,
    seed: int = 0,
    repeats: int = 3,
) -> MLXConformanceReport:
    """Run exact Apple-device checks; unavailable devices produce a falsy skip.

    ``backend='both'`` independently qualifies the compositional MLX and custom
    Metal paths. Array evaluation is explicit, so compilation and execution
    errors are captured in the report rather than deferred to the caller.
    """
    if backend not in (*MLX_BACKENDS, "auto", "both"):
        raise LocalizationError("backend must be 'both', 'auto', 'metal', or 'mlx'")
    batch = _positive_int("batch", batch)
    repeats = _positive_int("repeats", repeats)
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise LocalizationError("seed must be an integer")
    geometries = [dict(cell) for cell in (DEFAULT_MLX_MATRIX if matrix is None else matrix)]
    if not geometries:
        raise LocalizationError("matrix must contain at least one geometry")
    required = {
        "width",
        "block_size",
        "dcp_size",
        "dcp_rank",
        "dcp_interleave",
        "compact_valid_to_front",
    }
    for cell in geometries:
        cell.setdefault("compact_valid_to_front", True)
        if set(cell) != required:
            raise LocalizationError(f"matrix geometry must contain {', '.join(sorted(required))}")
        if _positive_int("width", cell["width"]) > MAX_ROW_WIDTH:
            raise LocalizationError(f"width must not exceed {MAX_ROW_WIDTH}")
        block_size = _positive_int("block_size", cell["block_size"])
        _validate_dcp_config(cell["dcp_size"], cell["dcp_rank"], cell["dcp_interleave"])
        if not isinstance(cell["compact_valid_to_front"], bool):
            raise LocalizationError("compact_valid_to_front must be a bool")
        for key in required - {"compact_valid_to_front"}:
            cell[key] = int(cell[key])
        if block_size % cell["dcp_interleave"]:
            raise LocalizationError("block_size must be divisible by dcp_interleave")
        _validate_fixture(cell["width"], batch, block_size, cell["dcp_size"], 17)
    mx, reason = _load_mlx()
    if reason:
        return MLXConformanceReport(skipped=reason)
    report = MLXConformanceReport(
        metadata={**_metadata(mx), "seed": int(seed), "batch": batch, "repeats": repeats}
    )
    arms = MLX_BACKENDS if backend == "both" else (backend,)
    for index, geometry in enumerate(geometries):
        for arm in arms:
            report.cells.append(
                _run_cell(mx, arm, geometry, batch=batch, seed=int(seed) + index, repeats=repeats)
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=(*MLX_BACKENDS, "auto", "both"), default="both")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--require-device", action="store_true", help="return exit 2 when Metal is unavailable"
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable report")
    args = parser.parse_args(argv)
    try:
        report = conformance_mlx(
            backend=args.backend, batch=args.batch, seed=args.seed, repeats=args.repeats
        )
    except LocalizationError as exc:
        parser.error(str(exc))
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.summary())
    if report.skipped:
        return 2 if args.require_device else 0
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
