"""Host-only CUDA launch validation: these tests never import or execute CUDA."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from math import prod
from types import SimpleNamespace

import pytest

from silkern import LocalizationError, localize_reference, workspace_shapes
from silkern import kernels as kernels_module
from silkern.kernels import _validate_launch_config


class _WrappingInt(int):
    """Model fixed-width Integral multiplication without a NumPy dependency."""

    def __mul__(self, other):
        return 0


def test_launch_configuration_normalizes_integrals_before_arithmetic() -> None:
    options = [_WrappingInt(value) for value in (64, 4, 3, 2)]
    normalized = _validate_launch_config(*options, True, _WrappingInt(4))
    assert normalized == (64, 4, 3, 2, 4)
    assert all(type(value) is int for value in normalized)
    with pytest.raises(LocalizationError, match="fit int32"):
        _validate_launch_config(65536, _WrappingInt(65536), 0, _WrappingInt(65536), True, 4)


def test_oracle_and_workspace_normalize_host_integer_scalars() -> None:
    expected = localize_reference(
        [0], [[4]], [[0, 1, 4, 5]], block_size=4, dcp_size=2, dcp_rank=0, dcp_interleave=2
    )
    observed = localize_reference(
        [0], [[4]], [[0, 1, 4, 5]], block_size=_WrappingInt(4),
        dcp_size=_WrappingInt(2), dcp_rank=_WrappingInt(0), dcp_interleave=_WrappingInt(2)
    )
    assert observed == expected
    with pytest.raises(LocalizationError, match="fit int32"):
        workspace_shapes(_WrappingInt(2**31), _WrappingInt(1))


@pytest.mark.parametrize("index,value", [
    (0, 2**31), (0, 4.0), (0, True), (0, 0),
    (1, 2**31), (1, False), (1, 0),
    (2, True), (2, 2), (2, -1),
    (3, 0), (3, 3), (4, 1), (5, 8.0), (5, True), (5, 16),
])
def test_unsupported_launch_scalars_fail_before_kernel_specialization(index, value) -> None:
    args = [64, 2, 0, 1, True, 4]
    args[index] = value
    with pytest.raises(LocalizationError):
        _validate_launch_config(*args)


class _TensorMetadata:
    """Enough tensor metadata to exercise host launchers without importing CUDA."""

    dtype = "int32"
    device = "cuda:1"
    is_cuda = True

    def __init__(self, shape, pointer):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.pointer = pointer
        self.storage_pointer = pointer
        self.negative_view = False

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return 4

    def is_neg(self):
        return self.negative_view

    def is_contiguous(self):
        return True

    def stride(self, dimension):
        value = 1
        for extent in self.shape[dimension + 1:]:
            value *= extent
        return value

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage_pointer)


@pytest.fixture
def cuda_launch_metadata(monkeypatch):
    state = SimpleNamespace(device="cuda:0", launches=[], fail_at=None)

    @contextmanager
    def device(target):
        before = state.device
        state.device = target
        try:
            yield
        finally:
            state.device = before

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                state.launches.append((self.name, state.device, grid, kwargs))
                if len(state.launches) == state.fail_at:
                    raise RuntimeError("simulated launch failure")
            return launch

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=_TensorMetadata,
        int32="int32",
        cuda=SimpleNamespace(device=device),
    ))
    monkeypatch.setattr(kernels_module, "triton", SimpleNamespace(
        next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        cdiv=lambda numerator, denominator: (numerator + denominator - 1) // denominator,
    ))
    for name in ("_rowwise_kernel", "_map_tiles_kernel", "_tile_prefix_kernel",
                 "_fill_output_kernel", "_scatter_tiles_kernel"):
        monkeypatch.setattr(kernels_module, name, Kernel(name), raising=False)
    return state


def _metadata_tensors(arm, tile_size=128):
    shapes = [(2,), (2, 17), (2, 129), (2, 129), (2,)]
    if arm == "hierarchical":
        shapes.extend(workspace_shapes(2, 129, tile_size=tile_size).values())
    return [_TensorMetadata(shape, (index + 1) * 4096) for index, shape in enumerate(shapes)]


def _metadata_launch(arm, *, compact=True, tile_size=128, tensors=None):
    tensors = _metadata_tensors(arm, tile_size) if tensors is None else tensors
    options = dict(block_size=64, dcp_size=2, dcp_rank=0, compact_valid_to_front=compact)
    if arm == "hierarchical":
        kernels_module.localize_hierarchical(*tensors, **options, tile_size=tile_size)
    else:
        kernels_module.localize_rowwise(*tensors, **options)


@pytest.mark.parametrize("arm,compact,launch_count", [
    ("rowwise", True, 1), ("rowwise", False, 1),
    ("hierarchical", True, 4), ("hierarchical", False, 2),
])
def test_launchers_use_tensor_device_and_restore_callers_device(
    cuda_launch_metadata, arm, compact, launch_count,
) -> None:
    _metadata_launch(arm, compact=compact)
    assert len(cuda_launch_metadata.launches) == launch_count
    assert all(device == "cuda:1" for _, device, _, _ in cuda_launch_metadata.launches)
    assert cuda_launch_metadata.device == "cuda:0"


@pytest.mark.parametrize("arm,fail_at", [
    ("rowwise", 1), ("hierarchical", 1), ("hierarchical", 2),
    ("hierarchical", 3), ("hierarchical", 4),
])
def test_launchers_restore_callers_device_after_launch_failure(
    cuda_launch_metadata, arm, fail_at,
) -> None:
    cuda_launch_metadata.fail_at = fail_at
    with pytest.raises(RuntimeError, match="simulated launch failure"):
        _metadata_launch(arm)
    assert cuda_launch_metadata.device == "cuda:0"


def test_hierarchical_normalizes_tile_size_before_triton_specialization(cuda_launch_metadata) -> None:
    _metadata_launch("hierarchical", tile_size=_WrappingInt(128))
    tile_launches = [options for _, _, _, options in cuda_launch_metadata.launches
                    if "TILE_SIZE" in options]
    assert len(tile_launches) == 2
    assert all(type(options["TILE_SIZE"]) is int for options in tile_launches)


@pytest.mark.parametrize("arm", ["rowwise", "hierarchical"])
@pytest.mark.parametrize("offset", [-4, 4])
def test_imported_overlapping_output_storage_rejected_before_launch(cuda_launch_metadata, arm, offset):
    tensors = _metadata_tensors(arm)
    # DLPack can import an overlapping slice as a separate storage whose base
    # differs from the input's. Pointer equality alone misses this overlap.
    tensors[3].pointer = tensors[2].pointer + offset
    tensors[3].storage_pointer = tensors[3].pointer
    with pytest.raises(LocalizationError, match="storage|overlap"):
        _metadata_launch(arm, tensors=tensors)
    assert cuda_launch_metadata.launches == []


@pytest.mark.parametrize("pair", [(2, 5), (5, 6), (4, 8), (0, 7)])
def test_hierarchical_imported_overlap_with_a_written_buffer_rejected(cuda_launch_metadata, pair):
    tensors = _metadata_tensors("hierarchical")
    first, second = (tensors[index] for index in pair)
    second.pointer = first.pointer + 4
    second.storage_pointer = second.pointer
    with pytest.raises(LocalizationError, match="storage|overlap"):
        _metadata_launch("hierarchical", tensors=tensors)
    assert cuda_launch_metadata.launches == []


@pytest.mark.parametrize("arm", ["rowwise", "hierarchical"])
def test_adjacent_disjoint_memory_spans_are_accepted(cuda_launch_metadata, arm):
    tensors = _metadata_tensors(arm)
    tensors[3].pointer = tensors[2].pointer + tensors[2].numel() * tensors[2].element_size()
    tensors[3].storage_pointer = tensors[3].pointer
    _metadata_launch(arm, tensors=tensors)
    assert cuda_launch_metadata.launches


@pytest.mark.parametrize("arm", ["rowwise", "hierarchical"])
@pytest.mark.parametrize("shared", ["overlapping", "one storage"])
def test_read_only_inputs_may_alias_each_other(cuda_launch_metadata, arm, shared):
    # The kernels only read the three inputs, so both launchers let them alias.
    tensors = _metadata_tensors(arm)
    if shared == "overlapping":
        tensors[0].pointer = tensors[2].pointer + 4
        tensors[0].storage_pointer = tensors[0].pointer
    else:
        tensors[1].pointer = tensors[0].pointer + tensors[0].numel() * tensors[0].element_size()
        tensors[1].storage_pointer = tensors[0].storage_pointer
    _metadata_launch(arm, tensors=tensors)
    assert cuda_launch_metadata.launches


@pytest.mark.parametrize("arm, pair", [
    ("rowwise", (3, 4)), ("rowwise", (2, 3)), ("rowwise", (0, 4)),
    ("hierarchical", (3, 4)), ("hierarchical", (5, 7)), ("hierarchical", (0, 6)),
])
def test_written_buffers_need_storage_of_their_own(cuda_launch_metadata, arm, pair):
    # Non-overlapping byte ranges are not enough once a buffer is written.
    tensors = _metadata_tensors(arm)
    first, second = (tensors[index] for index in pair)
    second.pointer = first.pointer + first.numel() * first.element_size()
    second.storage_pointer = first.storage_pointer
    with pytest.raises(LocalizationError, match="distinct storage"):
        _metadata_launch(arm, tensors=tensors)
    assert cuda_launch_metadata.launches == []


@pytest.mark.parametrize("other_index", [0, 1, 2, 3])
def test_rowwise_imported_counts_overlap_rejected(cuda_launch_metadata, other_index):
    tensors = _metadata_tensors("rowwise")
    tensors[4].pointer = tensors[other_index].pointer + 4
    tensors[4].storage_pointer = tensors[4].pointer
    with pytest.raises(LocalizationError, match="storage|overlap"):
        _metadata_launch("rowwise", tensors=tensors)
    assert cuda_launch_metadata.launches == []


@pytest.mark.parametrize("batch", [2**31 - 1, 2**31])
def test_rowwise_batch_grid_boundary_without_allocating(cuda_launch_metadata, batch):
    shapes = [(batch,), (1, 1), (batch, 1), (batch, 1), (batch,)]
    tensors = [_TensorMetadata(shape, (index + 1) * 2**50)
               for index, shape in enumerate(shapes)]
    if batch == 2**31:
        with pytest.raises(LocalizationError, match="batch.*int32"):
            _metadata_launch("rowwise", tensors=tensors)
        assert cuda_launch_metadata.launches == []
    else:
        _metadata_launch("rowwise", tensors=tensors)
        assert cuda_launch_metadata.launches[0][2] == (batch,)


@pytest.mark.parametrize("arm,index", [("rowwise", index) for index in range(5)]
                         + [("hierarchical", index) for index in range(9)])
def test_lazy_negation_view_rejected_before_raw_pointer_launch(cuda_launch_metadata, arm, index):
    tensors = _metadata_tensors(arm)
    tensors[index].negative_view = True
    with pytest.raises(LocalizationError, match="negation|negative view"):
        _metadata_launch(arm, tensors=tensors)
    assert cuda_launch_metadata.launches == []
