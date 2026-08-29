"""The benchmarks are not shipped, so nothing else imports them. This does.

``bench/`` is 500-odd lines that only ever run on a GPU box. Import-level rot
there is silent until someone tries to reproduce a number from the README, which
is the worst possible moment to discover it.
"""

from __future__ import annotations

import pytest

from bench import atomic_baseline, bench_converter, order_instability


@pytest.mark.parametrize("module", [bench_converter, order_instability])
def test_benchmark_entry_points_parse_their_arguments(module) -> None:
    assert callable(module.main)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0


def test_atomic_baseline_exposes_its_launcher() -> None:
    """The baseline is imported by name from both benchmarks and from the docs."""
    assert callable(atomic_baseline.launch_atomic_baseline)
