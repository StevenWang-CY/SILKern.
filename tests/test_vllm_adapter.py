from __future__ import annotations

from types import SimpleNamespace

import pytest

from silkern.contract import SUPPORTED_TILE_SIZES
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
        ("compact_valid_to_front", False),
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
