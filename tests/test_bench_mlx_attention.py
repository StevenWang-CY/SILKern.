"""Independent consumer-oracle, qualification-gate, and raw-evidence regressions."""

from __future__ import annotations

import copy
import inspect
import json
import math
import textwrap
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import bench_mlx_attention as attention
from silkern.errors import LocalizationError


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cells": []},
        {"cells": [(1, 128), (1, 128)]},
        {"cells": [(1,)]},
        {"cells": [(1, 1)]},
        {"cells": [(1, 4097)]},
        {"cells": [(True, 128)]},
        {"cells": [(1024, 128)]},
        {"warmup": 0},
        {"iterations": -1},
        {"blocks": False},
        {"seed": True},
    ],
)
def test_invalid_options_fail_before_runtime_load(monkeypatch, kwargs):
    monkeypatch.setattr(attention, "_load_mlx", lambda: pytest.fail("loaded MLX"))
    with pytest.raises(LocalizationError):
        attention.benchmark_attention(**kwargs)


@pytest.fixture(scope="module")
def fixture():
    return attention._fixture(2, 8, 11)


def test_reference_preserves_duplicate_multiplicity_and_empty_selections(fixture):
    _, requests, _, _, keys, values = fixture
    queries = [[0.0] * attention.KEY_DIM] * 2
    rows = [[0, 3, 3, -1, len(keys[0])], [-1] * 5]
    output, counts = attention._reference(queries, requests, rows, keys, values)
    assert counts == [[1, 0], [2, 0]]
    assert output[1] == [0.0] * attention.VALUE_DIM
    assert output[0] == pytest.approx(
        [
            math.fsum(values[requests[0]][token][dimension] for token in (0, 3, 3)) / 3
            for dimension in range(attention.VALUE_DIM)
        ]
    )


def test_paged_fixture_poisons_every_slot_no_token_occupies(fixture):
    _, requests, rows, shards, logical_keys, logical_values = fixture
    assert rows[-1] == [-1] * 8
    assert rows[0].count(3) == 2
    for rank, (table, keys, values) in enumerate(shards):
        pages = [page for row in table for page in row]
        assert len(set(pages)) == len(pages)
        assert pages != sorted(pages)
        occupied = {
            page * attention.BLOCK_SIZE + offset
            for page in pages
            for offset in range(attention.BLOCK_SIZE)
        }
        for slot, (key, value) in enumerate(zip(keys, values, strict=True)):
            finite = [math.isfinite(entry) for entry in key + value]
            # Placeholder page 0 and every other unowned slot is NaN throughout.
            assert all(finite) if slot in occupied else not any(finite), slot
        assert not any(slot < attention.BLOCK_SIZE for slot in occupied)
        for request in requests:
            for token in (rank, 2 + rank, 128 + rank):
                local = token // attention.SHARDS
                slot = (
                    table[request][local // attention.BLOCK_SIZE] * attention.BLOCK_SIZE
                    + local % attention.BLOCK_SIZE
                )
                assert keys[slot] == logical_keys[request][token]
                assert values[slot] == logical_values[request][token]


class _Array:
    def __init__(self, data, dtype=None):
        self.data = data

    def tolist(self):
        return self.data


@pytest.mark.parametrize("frozen,name", [(0, "query"), (1, "request_ids"), (2, "token_indices")])
def test_qualification_rejects_independently_frozen_inputs(fixture, frozen, name):
    query, requests, rows, shards, keys, values = fixture
    mx = SimpleNamespace(array=_Array, int32="int32", float32="float32", eval=lambda *args: None)
    baseline = None

    def broken(*arrays):
        nonlocal baseline
        if baseline is None:
            baseline = arrays
        arrays = list(arrays)
        arrays[frozen] = baseline[frozen]
        output, counts = attention._reference(*(array.data for array in arrays[:3]), keys, values)
        return _Array(output), tuple(_Array(row) for row in counts)

    arrays = tuple(_Array(data) for data in (query, requests, rows)) + (shards,)
    with pytest.raises(LocalizationError, match=f"broken/{name} consumer oracle mismatch"):
        attention._qualify_inputs(
            mx, {"broken": broken}, arrays, query, requests, rows, keys, values
        )


@pytest.mark.parametrize("wrong", ["counts", "nan", "shape", "value"])
def test_oracle_gate_rejects_incorrect_consumer_outputs(wrong):
    output, counts = [[0.5]], [[1], [0]]
    if wrong == "counts":
        counts = [[0], [1]]
    elif wrong == "nan":
        output = [[float("nan")]]
    elif wrong == "shape":
        output = [[]]
    else:
        output = [[0.6]]
    with pytest.raises(LocalizationError, match="consumer oracle mismatch"):
        attention._check_output("test", output, counts, [[0.5]], [[1], [0]])


def _session(index):
    return {
        "schema_version": 1,
        "benchmark": "silkern_mlx_selected_attention",
        "created_at": f"2026-10-02T03:00:0{index}+00:00",
        "metadata": {
            "platform": "test",
            "machine": "arm64",
            "python": "3.12",
            "versions": {"silkern": "0.2.0", "mlx": "0.32.3"},
            "device": {"device_name": "Test"},
            "execution_backend": "apple_metal",
            "source_sha256": dict.fromkeys(attention._SOURCE_FILES, "a" * 64),
        },
        "methodology": {
            **copy.deepcopy(attention._PROTOCOL),
            "warmup": 10,
            "iterations_per_block": 50,
            "blocks": 3,
            "seed": 11,
        },
        "cells": [
            {
                "geometry": {
                    "batch": 1,
                    "width": 128,
                    "block_size": 64,
                    "dcp_size": 2,
                    "dcp_interleave": 1,
                    "key_dim": 64,
                    "value_dim": 64,
                    "request_count": 4,
                    "table_width": 2,
                },
                "seed": 11,
                "oracle_passed": True,
                "changing_inputs_passed": True,
                "valid_counts": [[63], [64]],
                "oracle_max_abs_error": dict.fromkeys(attention.ARMS, 0.0),
                "schedule": [
                    list(attention.ARMS[block % 2 :] + attention.ARMS[: block % 2])
                    for block in range(3)
                ],
                "results": {
                    arm: {
                        "block_samples_us": [
                            scale + index,
                            3 * (scale + index),
                            99 * (scale + index),
                        ],
                        "median_us": -1,
                    }
                    for arm, scale in zip(attention.ARMS, (10, 5), strict=True)
                },
            }
        ],
    }


def _write(tmp_path, sessions):
    paths = []
    for index, session in enumerate(sessions):
        path = tmp_path / f"session-{index}.json"
        path.write_text(json.dumps(session))
        paths.append(path)
    return paths


def test_summary_recomputes_raw_medians_and_cli_requires_no_device(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(attention, "_load_mlx", lambda: pytest.fail("loaded MLX"))
    paths = _write(tmp_path, [_session(index) for index in range(3)])
    summary = attention.summarize_attention(paths)
    assert summary["cells"][0]["median_us"] == {"mlx_compiled": 33, "metal_compiled": 18}
    assert summary["cells"][0]["compiled_ratio"] == 33 / 18
    assert attention.main(["--summarize", *map(str, paths)]) == 0
    assert json.loads(capsys.readouterr().out) == summary
    before = paths[0].read_bytes()
    assert attention.main(["--summarize", *map(str, paths), "--output", str(paths[0])]) == 2
    assert paths[0].read_bytes() == before


@pytest.mark.parametrize(
    "kind",
    [
        "source",
        "device",
        "version",
        "counts",
        "negative_count",
        "count_total",
        "errors",
        "raw",
        "schedule",
        "huge_blocks",
        "geometry",
        "oracle",
        "changing",
        "method_bool",
        "duplicate",
    ],
)
def test_summary_rejects_invalid_or_unqualified_matching_records(tmp_path, kind):
    sessions = [_session(0), _session(1)]
    for session in sessions:
        cell = session["cells"][0]
        if kind == "source":
            session["metadata"]["source_sha256"].pop("examples/mlx_sparse_attention.py")
        elif kind == "device":
            session["metadata"]["device"] = {}
        elif kind == "version":
            session["metadata"]["versions"].pop("mlx")
        elif kind == "counts":
            cell["valid_counts"] = 128
        elif kind == "negative_count":
            cell["valid_counts"][0][0] = -1
        elif kind == "count_total":
            cell["valid_counts"] = [[128], [128]]
        elif kind == "errors":
            cell["oracle_max_abs_error"]["mlx_compiled"] = -1
        elif kind == "raw":
            cell["results"]["mlx_compiled"]["block_samples_us"][0] = 0
        elif kind == "schedule":
            cell["schedule"] = [list(attention.ARMS)] * 3
        elif kind == "huge_blocks":
            session["methodology"]["blocks"] = 2**62
        elif kind == "geometry":
            cell["geometry"]["key_dim"] = 32
        elif kind == "oracle":
            cell["oracle_passed"] = False
        elif kind == "changing":
            cell["changing_inputs_passed"] = 1
        elif kind == "method_bool":
            session["methodology"]["full_model_decode"] = 0
    if kind == "duplicate":
        sessions[1]["cells"] = copy.deepcopy(sessions[0]["cells"])
    with pytest.raises(ValueError):
        attention.summarize_attention(_write(tmp_path, sessions))


@pytest.mark.mlx
def test_gate_rejects_a_consumer_that_skips_masking_gathered_values():
    """Padded slots read NaN, so an unmasked consumer cannot pass the oracle gate."""
    mx = pytest.importorskip("mlx.core")
    example = pytest.importorskip("examples.mlx_sparse_attention")
    source = inspect.getsource(example.selected_attention)
    masks = (
        "gathered_keys = mx.where(valid[..., None], gathered_keys, 0)",
        "gathered_values = mx.where(valid[..., None], gathered_values, 0)",
    )
    assert all(mask in source for mask in masks), "the example changed; update this test"
    for mask in masks:
        source = source.replace(mask, "pass")
    namespace = dict(vars(example))
    exec(compile(textwrap.dedent(source), "unmasked_consumer", "exec"), namespace)
    stream = mx.default_stream(mx.cpu)
    query, requests, rows, shards, keys, values = attention._fixture(2, 8, 11)
    arrays = (
        mx.array(query, dtype=mx.float32),
        mx.array(requests, dtype=mx.int32),
        mx.array(rows, dtype=mx.int32),
        [
            (mx.array(table, dtype=mx.int32), mx.array(k, dtype=mx.float32),
             mx.array(v, dtype=mx.float32))
            for table, k, v in shards
        ],
    )
    options = dict(block_size=attention.BLOCK_SIZE, backend="mlx", stream=stream)
    unmasked = {"unmasked": partial(namespace["selected_attention"], **options)}
    with pytest.raises(LocalizationError, match="attention values"):
        attention._qualify_inputs(mx, unmasked, arrays, query, requests, rows, keys, values)
    masked = {"masked": partial(example.selected_attention, **options)}
    errors = attention._qualify_inputs(mx, masked, arrays, query, requests, rows, keys, values)
    assert errors["masked"] < 1e-5


@pytest.mark.mlx
@pytest.mark.metal
def test_actual_compiled_consumer_qualifies_before_measurement(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Apple Metal unavailable")
    measured = []

    def measure(mx, arms, **kwargs):
        measured.append(set(arms))
        return {arm: [1.0] for arm in arms}, [list(arms)]

    monkeypatch.setattr(attention, "_measure", measure)
    report = attention.benchmark_attention(cells=[(2, 8)], warmup=1, iterations=1, blocks=1)
    assert measured == [set(attention.ARMS)]
    assert report["cells"][0]["oracle_passed"]
    assert report["cells"][0]["changing_inputs_passed"]
    assert not report["methodology"]["full_model_decode"]


def test_retained_consumer_summary_reproduces_exactly():
    root = Path(__file__).resolve().parents[1] / "evidence" / "09-apple-mlx-consumer"
    paths = sorted(root.glob("consumer-session-*.json"))
    assert len(paths) == 3
    assert attention.summarize_attention(paths) == json.loads((root / "consumer-summary.json").read_text())
