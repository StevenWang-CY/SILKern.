"""Seeded differential fuzzing of the MLX and Metal backends against the oracle.

Targeted regressions live in ``test_mlx.py``; this module draws whole random
geometries instead -- widths around every scan boundary, non-power-of-two
interleaves and page sizes, negative and out-of-table pages, foreign and
negative tokens, duplicates, and out-of-range request ids -- and requires exact
agreement with ``localize_reference`` on every execution path.

``SILKERN_MLX_FUZZ_CASES`` raises the number of cases per path (default 48).
"""

from __future__ import annotations

import os
import random

import pytest

from silkern import localize_mlx, localize_reference

pytestmark = pytest.mark.mlx
mx = pytest.importorskip("mlx.core")

CASES = int(os.environ.get("SILKERN_MLX_FUZZ_CASES", "48"))
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
# Widths that straddle SIMD-group (32), threadgroup (256) and items-per-thread
# boundaries of the Metal kernel, plus the accelerator row limit.
BOUNDARY_WIDTHS = (1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513,
                   1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096)


@pytest.fixture(params=[
    "mlx_cpu",
    pytest.param("mlx_gpu", marks=pytest.mark.metal),
    pytest.param("metal", marks=pytest.mark.metal),
])
def execution(request):
    if request.param != "mlx_cpu" and not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    device = mx.cpu if request.param == "mlx_cpu" else mx.gpu
    with mx.stream(device):
        yield "metal" if request.param == "metal" else "mlx", device


def _random_case(rng: random.Random) -> dict:
    interleave = rng.choice((1, 1, 2, 3, 4, 8))
    block_size = interleave * rng.choice((1, 2, 3, 5, 8, 16, 64))
    dcp_size = rng.choice((1, 1, 2, 3, 4, 8))
    requests, table_width = rng.randint(1, 4), rng.randint(1, 8)
    # Pages stay below 2**20 so every translated slot fits int32.
    table = [[rng.randrange(-4, 2**20) for _ in range(table_width)] for _ in range(requests)]
    width = rng.choice(BOUNDARY_WIDTHS) if rng.random() < 0.5 else rng.randint(1, 4096)
    covered = block_size * table_width * dcp_size
    specials = (INT32_MIN, -1, 0, INT32_MAX, covered - 1, covered)
    rows = []
    for _ in range(rng.randint(1, 5)):
        density = rng.random()  # from nearly empty to nearly full selections
        row: list[int] = []
        for _ in range(width):
            draw = rng.random()
            if draw < 0.03:
                row.append(rng.choice(specials))
            elif draw < 0.1 and row:
                row.append(rng.choice(row))
            elif draw < density:
                row.append(rng.randrange(covered))
            elif draw < density + 0.1:
                row.append(rng.randrange(-50, 0))
            else:
                row.append(rng.randrange(covered, min(INT32_MAX, 4 * covered) + 1))
        rows.append(row)
    req_ids = [
        rng.choice((-1, requests, INT32_MIN, INT32_MAX)) if rng.random() < 0.15
        else rng.randrange(requests)
        for _ in rows
    ]
    return dict(req_ids=req_ids, table=table, rows=rows, block_size=block_size,
                dcp_size=dcp_size, dcp_rank=rng.randrange(dcp_size),
                dcp_interleave=interleave, compact_valid_to_front=rng.random() < 0.7)


def _expected(case: dict) -> tuple[list[list[int]], list[int]]:
    """Oracle per row; MLX masks out-of-range request ids to an empty row."""
    config = {key: case[key] for key in
              ("block_size", "dcp_size", "dcp_rank", "dcp_interleave", "compact_valid_to_front")}
    out, counts = [], []
    for request, row in zip(case["req_ids"], case["rows"], strict=True):
        if not 0 <= request < len(case["table"]):
            out.append([-1] * len(row))
            counts.append(0)
            continue
        (row_out,), (count,) = localize_reference([request], case["table"], [row], **config)
        out.append(row_out)
        counts.append(count)
    return out, counts


def test_random_geometries_match_the_oracle_exactly(execution) -> None:
    backend, stream = execution
    rng = random.Random(20261007)
    for index in range(CASES):
        case = _random_case(rng)
        inputs = [mx.array(case[key], dtype=mx.int32) for key in ("req_ids", "table", "rows")]
        out, counts = localize_mlx(
            *inputs, block_size=case["block_size"], dcp_size=case["dcp_size"],
            dcp_rank=case["dcp_rank"], dcp_interleave=case["dcp_interleave"],
            compact_valid_to_front=case["compact_valid_to_front"],
            backend=backend, stream=stream,
        )
        mx.eval(out, counts)
        width = len(case["rows"][0])
        label = (f"case {index}: width={width} block={case['block_size']} "
                 f"dcp={case['dcp_size']}/{case['dcp_rank']} interleave={case['dcp_interleave']}")
        assert (out.tolist(), counts.tolist()) == _expected(case), label
        assert [array.tolist() for array in inputs] == [
            case["req_ids"], case["table"], case["rows"]
        ], f"{label}: inputs mutated"


def test_repeated_evaluation_is_bitwise_identical(execution) -> None:
    """The same inputs give the same bytes on every evaluation, not just the same set."""
    backend, stream = execution
    rng = random.Random(7)
    for index in range(max(4, CASES // 8)):
        case = _random_case(rng)
        inputs = [mx.array(case[key], dtype=mx.int32) for key in ("req_ids", "table", "rows")]
        config = dict(block_size=case["block_size"], dcp_size=max(2, case["dcp_size"]),
                      dcp_rank=0, dcp_interleave=case["dcp_interleave"], backend=backend,
                      stream=stream)
        results = []
        for _ in range(5):
            out, counts = localize_mlx(*inputs, **config)
            mx.eval(out, counts)
            results.append((out.tolist(), counts.tolist()))
        assert all(result == results[0] for result in results), f"case {index}"
