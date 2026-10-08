from __future__ import annotations

from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from silkern import localize_reference  # noqa: E402
from silkern.integrations.vllm import (  # noqa: E402
    AdapterError,
    WorkspaceAdapter,
)

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

#: Fills every written buffer before a replay; no correct launch writes it.
POISON = -(2**31)


def _case(variant: int = 0):
    """A qualified-surface batch; another ``variant`` changes every input."""
    batch = 3
    width = 2048
    table_width = 32
    req_ids = [(request + variant) % 3 for request in (2, 0, 1)]
    block_table = [
        [request * table_width + (column * 7 + variant) % table_width
         for column in range(table_width)]
        for request in range(3)
    ]
    rows = []
    for row in range(batch):
        values = [
            -1 if column % 19 == variant else (column * 13 + row * 17 + variant * 5) % 4096
            for column in range(width)
        ]
        rows.append(values)
    return req_ids, block_table, rows


def _grows_memory(op: Callable[[], None]) -> bool:
    """Whether ``op`` raised allocated device memory, at its end or at its peak."""
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    op()
    return torch.cuda.memory_allocated() > before or torch.cuda.max_memory_allocated() > before


@pytest.mark.parametrize("arm", ["row_stable", "hierarchical_stable"])
def test_adapter_matches_exact_oracle_and_replays_at_fixed_addresses(arm: str) -> None:
    cases = [_case(variant) for variant in (0, 1)]
    expected = [
        localize_reference(*case, block_size=64, dcp_size=2, dcp_rank=1, dcp_interleave=1)
        for case in cases
    ]
    assert expected[0] != expected[1]  # so a graph replaying stale inputs cannot pass
    staged = [
        tuple(torch.tensor(values, dtype=torch.int32, device="cuda") for values in case)
        for case in cases
    ]
    req_ids, block_table, tokens = (tensor.clone() for tensor in staged[0])
    adapter = WorkspaceAdapter(arm)  # type: ignore[arg-type]
    adapter.prepare(
        req_ids,
        block_table,
        tokens,
        dcp_size=2,
        dcp_rank=1,
        block_size=64,
    )
    signatures = adapter.fixed_buffer_signatures

    def op() -> None:
        output, counts = adapter(
            req_ids,
            block_table,
            tokens,
            2,
            1,
            BLOCK_SIZE=64,
            NUM_TOPK_TOKENS=2048,
            BLOCK_N=128,
            return_valid_counts=True,
        )
        assert output.data_ptr() == adapter.output.data_ptr()
        assert counts.data_ptr() == adapter.counts.data_ptr()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            assert not _grows_memory(op)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Measured inside the capture: beginning one may allocate PyTorch's
        # own bookkeeping, which is no launcher's doing.
        captured_grows = _grows_memory(op)
    assert not captured_grows
    written = (adapter.output, adapter.counts, *adapter._binding.workspace.values())
    for replay in range(100):
        # Rewrite the bound inputs in place, starting with the case the graph
        # did not capture, and poison every written buffer: a replay that does
        # no work, or reads inputs frozen at capture, cannot match its case.
        index = (replay + 1) % 2
        for bound, values in zip((req_ids, block_table, tokens), staged[index], strict=True):
            bound.copy_(values)
        for buffer in written:
            buffer.fill_(POISON)
        assert not _grows_memory(graph.replay)
        torch.cuda.synchronize()
        assert (adapter.output.cpu().tolist(), adapter.counts.cpu().tolist()) == expected[index]
    assert adapter.fixed_buffer_signatures == signatures


def test_adapter_rejects_a_different_fixed_address_binding() -> None:
    case = _case()
    req_ids, block_table, tokens = (
        torch.tensor(values, dtype=torch.int32, device="cuda") for values in case
    )
    adapter = WorkspaceAdapter("row_stable")
    adapter.prepare(
        req_ids,
        block_table,
        tokens,
        dcp_size=2,
        dcp_rank=0,
        block_size=64,
    )
    with pytest.raises(AdapterError, match="binding changed"):
        adapter(
            req_ids.clone(),
            block_table,
            tokens,
            2,
            0,
            BLOCK_SIZE=64,
            NUM_TOPK_TOKENS=2048,
            return_valid_counts=True,
        )
