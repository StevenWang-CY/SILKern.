"""The benchmarks are not shipped, so nothing else imports them. This does.

``bench/`` is 500-odd lines that only ever run on a GPU box. Import-level rot
there is silent until someone tries to reproduce a number from the README, which
is the worst possible moment to discover it.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from bench import atomic_baseline, bench_converter, order_instability
from silkern import LocalizationError, localize_reference


@pytest.mark.parametrize("module", [bench_converter, order_instability])
def test_benchmark_entry_points_parse_their_arguments(module) -> None:
    assert callable(module.main)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0


def test_atomic_baseline_exposes_its_launcher() -> None:
    """The baseline is imported by name from both benchmarks and from the docs."""
    assert callable(atomic_baseline.launch_atomic_baseline)


@pytest.mark.parametrize("module", [bench_converter, order_instability])
@pytest.mark.parametrize("args", [
    ["--width", "0"], ["--width", "4097"], ["--batch", "0"],
    ["--block-size", "0"], ["--dcp-size", "0"], ["--dcp-rank", "2"],
    ["--dcp-rank", "-1"], ["--replays", "0"], ["--replays", "-1"],
    ["--batch", "100000000"], ["--block-size", "2147483647"],
    ["--dcp-size", "2147483647"],
])
def test_invalid_diagnostics_fail_before_loading_device(module, args, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: pytest.fail("device runtime was accessed")
    )))
    with pytest.raises(SystemExit) as excinfo:
        module.main(args)
    assert excinfo.value.code == 2


@pytest.mark.parametrize("args", [
    ["--blocks", "0"], ["--blocks", "-1"], ["--tile-size", "32"],
    ["--with-atomic", "--dcp-size", "1"],
])
def test_converter_rejects_bad_sampling_and_atomic_geometry(args, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: pytest.fail("device runtime was accessed")
    )))
    with pytest.raises(SystemExit) as excinfo:
        bench_converter.main(args)
    assert excinfo.value.code == 2


def test_order_diagnostic_rejects_noncompacting_single_rank(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit) as excinfo:
        order_instability.main(["--dcp-size", "1"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("width,block_size,dcp_size", [(4096, 1, 1), (4096, 1, 2), (129, 32, 4)])
def test_fixture_supports_page_tables_larger_than_original_population(width, block_size, dcp_size) -> None:
    args = (width, 2, block_size, dcp_size, 13)
    req, table, rows = bench_converter._case(*args)
    assert (req, table, rows) == bench_converter._case(*args)
    assert len(rows) == 2
    assert all(len(row) == width for row in rows)
    for rank in range(dcp_size):
        out, counts = localize_reference(req, table, rows, block_size=block_size,
                                         dcp_size=dcp_size, dcp_rank=rank)
        assert all(0 <= count <= width for count in counts)
        assert all(-1 <= value < 2**31 for row in out for value in row)


def test_direct_fixture_generation_rejects_unbounded_host_allocations() -> None:
    with pytest.raises(LocalizationError, match="diagnostic fixtures"):
        bench_converter._case(4096, 1, 1, 4096, 0)


@pytest.mark.parametrize("name", ["row_stable", "hierarchical_stable", "silkern row_stable"])
def test_stable_qualification_requires_exact_order(name) -> None:
    bench_converter._check_results(name, [[7, 2, -1]], [2], [[7, 2, -1]], [2])
    with pytest.raises(LocalizationError, match="output values or order"):
        bench_converter._check_results(name, [[2, 7, -1]], [2], [[7, 2, -1]], [2])


def test_atomic_qualification_accepts_only_multiset_permutations() -> None:
    # Negative mapped values (including -1) may be valid; counts define the prefix.
    expected = [[-1, 7, 7, 2, -1, -1]]
    bench_converter._check_results("atomic_baseline", [[7, 2, -1, 7, -1, -1]], [4], expected, [4])
    for output, counts in (
        ([[7, 2, -1, 7, -1, -1]], [3]),
        ([[7, 2, -1, 2, -1, -1]], [4]),  # Same set, incorrect multiplicities.
        ([[7, 2, -1, 7, 0, -1]], [4]),
        ([[7, 2, -1, 7, -1]], [4]),
        ([], [4]),
    ):
        with pytest.raises(LocalizationError, match="oracle mismatch"):
            bench_converter._check_results("atomic_baseline", output, counts, expected, [4])


@pytest.mark.parametrize("atomic_orders,fragment", [
    ({"a"}, "no atomic order variation observed"),
    ({"a", "b"}, "atomic output order varied"),
])
def test_order_diagnostic_summary_reflects_observations(atomic_orders, fragment) -> None:
    observations = {
        "atomic_baseline": (atomic_orders, {"set"}, {"count"}),
        "silkern row_stable": ({"stable"}, {"set"}, {"count"}),
        "silkern hierarchical": ({"stable"}, {"set"}, {"count"}),
    }
    code, summary = order_instability._summarize_observations(observations)
    assert code == 0
    assert fragment in summary


@pytest.mark.parametrize("bad_result", [
    ({"a", "b"}, {"set"}, {"count"}),
    ({"a"}, {"wrong set"}, {"count"}),
    ({"a"}, {"set"}, {"wrong count"}),
])
def test_order_diagnostic_fails_on_stable_or_cross_arm_disagreement(bad_result) -> None:
    code, summary = order_instability._summarize_observations({
        "atomic_baseline": ({"a"}, {"set"}, {"count"}),
        "silkern row_stable": bad_result,
    })
    assert code == 1
    assert "FAIL" in summary
