from __future__ import annotations

import gc
import sys
import weakref
from contextlib import contextmanager
from math import prod
from types import SimpleNamespace

import pytest

from silkern.contract import SUPPORTED_TILE_SIZES
from silkern.integrations import vllm
from silkern.integrations.vllm import (
    AdapterError,
    WorkspaceAdapter,
    _tensor_signature,
    install_converter,
    validate_production_call,
)


def _valid_call() -> dict:
    return {
        "dcp_size": 2,
        "dcp_rank": 0,
        "cp_kv_cache_interleave_size": 1,
        "block_size": 64,
        "num_topk_tokens": 2048,
        "block_n": 128,
        "return_valid_counts": True,
        "compact_valid_to_front": True,
    }


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
def test_integrated_surface_accepts_pinned_backend_geometry(
    dcp_size: int,
    block_size: int,
) -> None:
    values = _valid_call()
    values["dcp_size"] = dcp_size
    values["dcp_rank"] = dcp_size - 1
    values["block_size"] = block_size
    validate_production_call(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dcp_size", 1),
        ("dcp_size", 8),
        ("dcp_rank", -1),
        ("dcp_rank", 2),
        ("dcp_rank", True),
        ("cp_kv_cache_interleave_size", 2),
        ("block_size", 16),
        ("block_size", 128),
        ("num_topk_tokens", 1024),
        ("block_n", 64),
        ("return_valid_counts", False),
        ("return_valid_counts", 1),
        ("return_valid_counts", "yes"),
        ("compact_valid_to_front", False),
        ("compact_valid_to_front", 1),
    ],
)
def test_integrated_surface_fails_closed(field: str, value: object) -> None:
    values = _valid_call()
    values[field] = value
    with pytest.raises(AdapterError):
        validate_production_call(**values)


def test_adapter_requires_known_arm() -> None:
    with pytest.raises(AdapterError):
        WorkspaceAdapter("atomic")  # type: ignore[arg-type]


@pytest.mark.parametrize("tile_size", [0, 1, 100, 512, True])
def test_adapter_rejects_an_unsupported_tile_size_at_construction(
    tile_size: object,
) -> None:
    """Better here than four frames deep in prepare(), on the real tensors."""
    with pytest.raises(AdapterError):
        WorkspaceAdapter("hierarchical_stable", hierarchical_tile_size=tile_size)


@pytest.mark.parametrize("tile_size", SUPPORTED_TILE_SIZES)
def test_adapter_accepts_every_supported_tile_size(tile_size: int) -> None:
    assert (
        WorkspaceAdapter(
            "hierarchical_stable", hierarchical_tile_size=tile_size
        ).hierarchical_tile_size
        == tile_size
    )


@pytest.mark.parametrize("value", [None, object(), 7, [1, 2, 3]])
def test_binding_signature_fails_closed_on_a_non_tensor(value: object) -> None:
    """This runs on the per-step path; an AttributeError there is not a failure
    mode the caller can distinguish from a real adapter rejection."""
    with pytest.raises(AdapterError, match="expected a tensor"):
        _tensor_signature(value)


def test_call_site_installation_is_scoped_and_restored() -> None:
    native = object()
    module = SimpleNamespace(triton_filter_and_convert_dcp_index=native)
    adapter = WorkspaceAdapter("row_stable")
    with install_converter(module, adapter):
        assert module.triton_filter_and_convert_dcp_index is adapter
    assert module.triton_filter_and_convert_dcp_index is native


def test_call_site_installation_rejects_unknown_source_surface() -> None:
    with pytest.raises(AdapterError):
        with install_converter(SimpleNamespace(), WorkspaceAdapter(
            "row_stable"
        )):
            pass


class _MetadataTensor:
    """Metadata-only tensor double: no CUDA runtime or device work is involved."""

    def __init__(self, shape, pointer):
        self.shape = tuple(shape)
        self.pointer = pointer
        self.dtype = "int32"
        self.device = "cuda:test"
        self.is_cuda = True
        self.storage_pointer = pointer
        self.negative_view = False

    @property
    def ndim(self):
        return len(self.shape)

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return 4

    def is_neg(self):
        return self.negative_view

    def stride(self):
        stride = []
        current = 1
        for extent in reversed(self.shape):
            stride.append(current)
            current *= extent
        return tuple(reversed(stride))

    def is_contiguous(self):
        return True

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage_pointer)


@pytest.fixture
def metadata_backend(monkeypatch):
    state = SimpleNamespace(allocations=0, launches=0, capturing=False,
                            current_device="cuda:caller", capturing_devices=set())

    def empty(shape, **kwargs):
        state.allocations += 1
        return _MetadataTensor((shape,) if isinstance(shape, int) else shape,
                               (1000 + state.allocations) * 2**40)

    @contextmanager
    def device(target):
        before = state.current_device
        state.current_device = target
        try:
            yield
        finally:
            state.current_device = before

    fake_torch = SimpleNamespace(
        Tensor=_MetadataTensor,
        int32="int32",
        empty=empty,
        empty_like=lambda tensor: empty(tensor.shape),
        cuda=SimpleNamespace(
            device=device,
            is_current_stream_capturing=lambda: (
                state.capturing or state.current_device in state.capturing_devices
            ),
        ),
    )

    def launch(*args, **kwargs):
        state.launches += 1

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(vllm, "localize_rowwise", launch)
    monkeypatch.setattr(vllm, "localize_hierarchical", launch)
    return state


def _metadata_inputs(batch=1, table_shape=(1, 32)):
    return (
        _MetadataTensor((batch,), 2**40),
        _MetadataTensor(table_shape, 2**41),
        _MetadataTensor((batch, 2048), 3 * 2**40),
    )


@pytest.mark.parametrize("arm", ["row_stable", "hierarchical_stable"])
def test_adapter_metadata_lifecycle_retains_and_reuses_buffers(metadata_backend, arm) -> None:
    inputs = _metadata_inputs()
    input_refs = [weakref.ref(tensor) for tensor in inputs]
    adapter = WorkspaceAdapter(arm)
    adapter.prepare(*inputs, dcp_size=2, dcp_rank=0)
    allocations = metadata_backend.allocations
    before = adapter.fixed_buffer_signatures
    for _ in range(3):
        out, counts = adapter(*inputs, 2, 0, return_valid_counts=True)
        assert out is adapter.output
        assert counts is adapter.counts
    assert adapter.calls == metadata_backend.launches == 3
    assert metadata_backend.allocations == allocations
    assert adapter.fixed_buffer_signatures == before
    del inputs
    gc.collect()
    assert all(ref() is not None for ref in input_refs)


def test_preparing_existing_adapter_cannot_reallocate_captured_buffers(metadata_backend) -> None:
    inputs = _metadata_inputs()
    adapter = WorkspaceAdapter("row_stable")
    adapter.prepare(*inputs, dcp_size=2, dcp_rank=0)
    allocations = metadata_backend.allocations
    before = adapter.fixed_buffer_signatures
    with pytest.raises(AdapterError, match="already prepared"):
        adapter.prepare(*inputs, dcp_size=2, dcp_rank=0)
    assert metadata_backend.allocations == allocations
    assert adapter.fixed_buffer_signatures == before


@pytest.mark.parametrize("target", ["out", "counts", "mapped", "tile_counts"])
@pytest.mark.parametrize("change", ["pointer", "shape"])
def test_modified_output_or_scratch_binding_rejected_before_launch(metadata_backend, target, change) -> None:
    inputs = _metadata_inputs()
    adapter = WorkspaceAdapter("hierarchical_stable")
    adapter.prepare(*inputs, dcp_size=2, dcp_rank=0)
    buffer = {
        "out": adapter.output, "counts": adapter.counts, **adapter._binding.workspace
    }[target]
    if change == "pointer":
        buffer.pointer += 100
    else:
        buffer.shape = (999,)
    with pytest.raises(AdapterError, match="output or workspace.*binding changed"):
        adapter(*inputs, 2, 0, return_valid_counts=True)
    assert metadata_backend.launches == adapter.calls == 0


@pytest.mark.parametrize("field, value", [("arm", "row_stable"), ("hierarchical_tile_size", 64)])
def test_adapter_dispatch_configuration_is_read_only(field, value) -> None:
    adapter = WorkspaceAdapter("hierarchical_stable")
    with pytest.raises(AttributeError):
        setattr(adapter, field, value)


@pytest.mark.parametrize("batch, table_shape", [(0, (1, 32)), (1, (0, 32)), (1, (1, 0))])
def test_empty_geometry_rejected_before_allocation(metadata_backend, batch, table_shape) -> None:
    adapter = WorkspaceAdapter("row_stable")
    with pytest.raises(AdapterError, match="must be positive"):
        adapter.prepare(*_metadata_inputs(batch, table_shape), dcp_size=2, dcp_rank=0)
    assert metadata_backend.allocations == 0
    assert not adapter.prepared


def test_prepare_rejects_capture_before_allocating(metadata_backend) -> None:
    metadata_backend.capturing = True
    adapter = WorkspaceAdapter("row_stable")
    with pytest.raises(AdapterError, match="before CUDA graph capture"):
        adapter.prepare(*_metadata_inputs(), dcp_size=2, dcp_rank=0)
    assert metadata_backend.allocations == 0
    assert not adapter.prepared


@pytest.mark.parametrize("arm", ["row_stable", "hierarchical_stable"])
@pytest.mark.parametrize("shared", ["one storage", "overlapping"])
def test_inputs_may_alias_each_other(metadata_backend, arm, shared) -> None:
    # The launchers only read the inputs, so views of one packed buffer are fine.
    inputs = _metadata_inputs()
    if shared == "one storage":
        inputs[1].storage_pointer = inputs[0].storage_pointer
    else:
        inputs[1].pointer = inputs[2].pointer + 4
        inputs[1].storage_pointer = inputs[1].pointer
    adapter = WorkspaceAdapter(arm)
    adapter.prepare(*inputs, dcp_size=2, dcp_rank=0)
    adapter(*inputs, 2, 0, return_valid_counts=True)
    assert adapter.calls == metadata_backend.launches == 1


@pytest.mark.parametrize(
    "call",
    [
        {"dcp_size": 4, "dcp_rank": 0},
        {"dcp_size": 2, "dcp_rank": 1},
        {"dcp_size": 2, "dcp_rank": 0, "BLOCK_SIZE": 32},
    ],
    ids=["dcp_size", "dcp_rank", "block_size"],
)
def test_call_geometry_differing_from_registration_rejected_before_launch(
    metadata_backend, call
) -> None:
    # Each call is inside the qualified surface; only the registration differs.
    inputs = _metadata_inputs()
    adapter = WorkspaceAdapter("row_stable")
    adapter.prepare(*inputs, dcp_size=2, dcp_rank=0, block_size=64)
    with pytest.raises(AdapterError, match=r"call geometry .* differs from registered"):
        adapter(*inputs, return_valid_counts=True, **call)
    assert metadata_backend.launches == adapter.calls == 0
    adapter(*inputs, 2, 0, BLOCK_SIZE=64, return_valid_counts=True)
    assert metadata_backend.launches == adapter.calls == 1


def test_hierarchical_oversized_workspace_rejected_before_allocation(metadata_backend) -> None:
    adapter = WorkspaceAdapter("hierarchical_stable")
    with pytest.raises(AdapterError, match="fit int32"):
        adapter.prepare(*_metadata_inputs(batch=2**31 // 2048), dcp_size=2, dcp_rank=0)
    assert metadata_backend.allocations == 0
    assert not adapter.prepared


def test_installation_restores_original_after_exception() -> None:
    original = object()
    module = SimpleNamespace(triton_filter_and_convert_dcp_index=original)
    with pytest.raises(RuntimeError, match="caller failed"):
        with install_converter(module, WorkspaceAdapter("row_stable")):
            raise RuntimeError("caller failed")
    assert module.triton_filter_and_convert_dcp_index is original


@pytest.mark.parametrize("index", range(3))
def test_prepare_rejects_lazy_negation_inputs_before_allocation(metadata_backend, index):
    inputs = _metadata_inputs()
    inputs[index].negative_view = True
    adapter = WorkspaceAdapter("row_stable")
    with pytest.raises(AdapterError, match="negation|negative view"):
        adapter.prepare(*inputs, dcp_size=2, dcp_rank=0)
    assert metadata_backend.allocations == 0
    assert not adapter.prepared


@pytest.mark.parametrize("target", ["req", "table", "tokens", "out", "counts", "mapped"])
def test_changing_lazy_negation_flag_rejected_before_dispatch(metadata_backend, target):
    inputs = _metadata_inputs()
    adapter = WorkspaceAdapter("hierarchical_stable")
    adapter.prepare(*inputs, dcp_size=2, dcp_rank=0)
    buffers = dict(zip(("req", "table", "tokens"), inputs, strict=True))
    buffers.update(out=adapter.output, counts=adapter.counts, **adapter._binding.workspace)
    buffers[target].negative_view = True
    with pytest.raises(AdapterError, match="binding changed|negation"):
        adapter(*inputs, 2, 0, return_valid_counts=True)
    assert metadata_backend.launches == 0


@pytest.mark.parametrize("capturing_device", ["cuda:caller", "cuda:test"])
def test_prepare_rejects_capture_on_caller_or_target_device(metadata_backend, capturing_device):
    metadata_backend.capturing_devices.add(capturing_device)
    adapter = WorkspaceAdapter("row_stable")
    with pytest.raises(AdapterError, match="before CUDA graph capture"):
        adapter.prepare(*_metadata_inputs(), dcp_size=2, dcp_rank=0)
    assert metadata_backend.allocations == 0
    assert metadata_backend.current_device == "cuda:caller"
    assert not adapter.prepared


def test_rowwise_oversized_batch_rejected_before_allocation(metadata_backend):
    adapter = WorkspaceAdapter("row_stable")
    with pytest.raises(AdapterError, match="batch.*int32"):
        adapter.prepare(*_metadata_inputs(batch=2**31), dcp_size=2, dcp_rank=0)
    assert metadata_backend.allocations == 0
    assert not adapter.prepared
