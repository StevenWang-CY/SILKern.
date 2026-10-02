"""Recompute reported Apple performance from raw samples without a device."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bench.summarize_mlx import main, summarize_sessions


def _session(index):
    arms = ("mlx", "metal", "mlx_compiled", "metal_compiled")
    return {
        "schema_version": 1,
        "benchmark": "silkern_mlx_localization",
        "created_at": f"2026-10-02T03:00:0{index}+00:00",
        "metadata": {
            "platform": "test",
            "machine": "arm64",
            "python": "3.12",
            "versions": {"silkern": "0.2.0", "mlx": "0.32.3"},
            "device": {"device_name": "Test"},
            "execution_backend": "apple_metal",
            "source_sha256": {
                name: "a" * 64
                for name in (
                    "silkern/contract.py", "silkern/mlx.py", "silkern/mlx_verify.py",
                    "bench/bench_mlx.py",
                )
            },
        },
        "methodology": {
            "measurement": "synchronized end-to-end functional call",
            "units": "microseconds per call",
            "includes": ["Python dispatch", "output allocation", "execution", "synchronization"],
            "excludes": ["input generation", "oracle validation", "first-use compilation"],
            "order": "rotate arm order each block",
            "full_model_decode": False,
            "blocks": 3,
            "iterations_per_block": 10,
            "warmup": 2,
            "seed": 11,
            "include_compiled": True,
        },
        "cells": [
            {
                "geometry": {
                    "width": 7,
                    "batch": 1,
                    "block_size": 64,
                    "dcp_size": 2,
                    "dcp_rank": 0,
                    "dcp_interleave": 1,
                    "compact_valid_to_front": True,
                },
                "seed": 11,
                "oracle_passed": True,
                "valid_fraction": 0.5,
                "schedule": [list(arms[index:] + arms[:index]) for index in range(3)],
                "results": {
                    arm: {
                        "block_samples_us": [scale, scale * 99, scale * 3],
                        "median_us": -999,
                    }  # Deliberately wrong cached statistics.
                    for arm, scale in zip(
                        arms, (10 + index, 5 + index, 8 + index, 4 + index), strict=True
                    )
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


def test_aggregation_recomputes_session_medians_and_divides_aggregates(tmp_path, capsys):
    paths = _write(tmp_path, [_session(index) for index in range(3)])
    summary = summarize_sessions(paths)
    assert summary["session_count"] == 3
    cell = summary["cells"][0]
    assert cell["median_us"] == {"mlx": 33, "metal": 18, "mlx_compiled": 27, "metal_compiled": 15}
    assert cell["eager_ratio"] == 33 / 18
    assert summary["compiled_ratio_range"] == [27 / 15, 27 / 15]
    output = tmp_path / "summary.json"
    assert main([*map(str, paths), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == summary
    assert main(list(map(str, paths))) == 0
    assert json.loads(capsys.readouterr().out) == summary


@pytest.mark.parametrize("kind", ["path", "content", "timestamp", "retimestamped_samples"])
def test_duplicate_sessions_are_rejected(tmp_path, kind):
    first, second = _session(0), _session(1)
    if kind == "content":
        second = copy.deepcopy(first)
    elif kind == "retimestamped_samples":
        second["cells"] = copy.deepcopy(first["cells"])
        second["cells"][0]["results"]["mlx"]["median_us"] = 123  # Cached summaries are irrelevant.
    elif kind == "timestamp":
        second["created_at"] = first["created_at"]
    paths = _write(tmp_path, [first, second])
    if kind == "path":
        paths[1] = paths[0]
    with pytest.raises(ValueError, match="duplicate session"):
        summarize_sessions(paths)


@pytest.mark.parametrize("invalid", [None, "metadata", "cell", "false_check", "mixed_schema"])
def test_schema_two_requires_explicit_compiled_input_qualification(tmp_path, invalid):
    sessions = [_session(0), _session(1)]
    for session in sessions:
        session["schema_version"] = 2
        session["methodology"]["compiled_input_checks"] = ["request_ids", "block_table", "token_indices"]
        session["cells"][0]["compiled_changing_inputs_passed"] = True
    if invalid == "metadata":
        sessions[1]["methodology"]["compiled_input_checks"].pop()
    elif invalid == "cell":
        sessions[1]["cells"][0].pop("compiled_changing_inputs_passed")
    elif invalid == "false_check":
        sessions[1]["cells"][0]["compiled_changing_inputs_passed"] = 1
    elif invalid == "mixed_schema":
        sessions[1]["schema_version"] = 1
    paths = _write(tmp_path, sessions)
    if invalid:
        with pytest.raises(ValueError):
            summarize_sessions(paths)
    else:
        assert summarize_sessions(paths)["session_count"] == 2


@pytest.mark.parametrize(
    "kind", ["geometry", "arms", "hash", "environment", "methodology", "raw", "oracle"]
)
def test_mixed_or_invalid_measurements_are_rejected(tmp_path, kind):
    first, second = _session(0), _session(1)
    if kind == "geometry":
        second["cells"][0]["geometry"]["width"] = 8
    elif kind == "arms":
        second["cells"][0]["results"].pop("metal_compiled")
    elif kind == "hash":
        second["metadata"]["source_sha256"]["silkern/mlx.py"] = "b" * 64
    elif kind == "environment":
        second["metadata"]["device"]["device_name"] = "Different Mac"
    elif kind == "methodology":
        second["methodology"]["iterations_per_block"] = 20
    elif kind == "raw":
        second["cells"][0]["results"]["mlx"]["block_samples_us"][0] = 0
    elif kind == "oracle":
        second["cells"][0]["oracle_passed"] = False
    with pytest.raises(ValueError):
        summarize_sessions(_write(tmp_path, [first, second]))


@pytest.mark.parametrize(
    "kind", ["source", "versions", "device", "platform", "method", "schedule", "full_model"]
)
def test_matching_sessions_still_require_complete_provenance_and_timing(tmp_path, kind):
    sessions = [_session(0), _session(1)]
    for session in sessions:
        if kind == "source":
            session["metadata"]["source_sha256"].pop("silkern/mlx.py")
        elif kind == "versions":
            session["metadata"]["versions"].pop("mlx")
        elif kind == "device":
            session["metadata"]["device"] = {}
        elif kind == "platform":
            session["metadata"]["platform"] = ""
        elif kind == "method":
            session["methodology"].pop("includes")
        elif kind == "schedule":
            cell = session["cells"][0]
            cell["schedule"] = [cell["schedule"][0]] * session["methodology"]["blocks"]
        elif kind == "full_model":
            session["methodology"]["full_model_decode"] = 0
    with pytest.raises(ValueError):
        summarize_sessions(_write(tmp_path, sessions))


def test_cli_does_not_overwrite_source_session(tmp_path, capsys):
    paths = _write(tmp_path, [_session(0), _session(1)])
    before = paths[0].read_bytes()
    assert main([*map(str, paths), "--output", str(paths[0])]) == 2
    assert paths[0].read_bytes() == before
    assert "must not overwrite" in capsys.readouterr().err


def test_cli_does_not_overwrite_source_session_through_a_hard_link(tmp_path, capsys):
    paths = _write(tmp_path, [_session(0), _session(1)])
    alias = tmp_path / "summary.json"
    alias.hardlink_to(paths[0])
    before = paths[0].read_bytes()
    assert main([*map(str, paths), "--output", str(alias)]) == 2
    assert paths[0].read_bytes() == before
    assert "must not overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("record", ["07-apple-mlx", "08-apple-mlx-audit", "09-apple-mlx-consumer"])
def test_retained_apple_summary_reproduces_exactly(record):
    evidence = Path(__file__).resolve().parents[1] / "evidence" / record
    paths = sorted(evidence.glob("benchmark-session-*.json"))
    assert len(paths) == 3
    assert summarize_sessions(paths) == json.loads((evidence / "summary.json").read_text())
