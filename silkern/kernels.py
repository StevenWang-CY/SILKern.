"""Allocation-free Triton implementations of the localization contract.

Two implementations, same contract, different dispatch envelopes:

``localize_rowwise``
    One program scans one whole selection row. The stable prefix falls out of a
    single row-wide ``cumsum`` -- no cross-program communication, no atomics,
    one kernel launch. Historical measurements distinguish converter cost from
    complete-step behavior; see ``docs/dispatch.md`` before choosing an arm.

``localize_hierarchical``
    Bounds each program's scan to one tile, composes deterministic tile-prefix
    offsets through caller-owned workspace, and performs a stable scatter. Four
    launches instead of one, with a bounded per-program working set. Measure
    both implementations in the intended consumer to select a dispatch policy.

Neither launcher allocates device memory. Every buffer -- inputs, outputs, and
workspace -- is supplied by the caller, so the same addresses can be captured
in a CUDA graph and reused while the bindings remain valid and buffer reuse is
ordered after consumer reads. Both validate metadata and raise
:class:`~silkern.errors.LocalizationError` rather than degrade.

``torch`` is imported lazily inside the launchers. Importing ``silkern`` does not
require PyTorch; a guarded Triton import defines kernels when it is installed
and importable. If it is absent or broken, the launchers raise and name why.
"""

from __future__ import annotations

from numbers import Integral

from silkern.contract import (
    DEFAULT_TILE_SIZE,
    MAX_ROW_WIDTH,
    _validate_compact_flag,
    _validate_dcp_config,
)
from silkern.errors import LocalizationError
from silkern.workspace import workspace_shapes


def _import_failure(package: str, exc: Exception) -> str:
    """Say why an optional runtime is unusable, for errors and skip reports.

    A missing package raises ``ModuleNotFoundError`` naming itself. An installed
    but broken one can raise anything at import -- ``ImportError`` or ``OSError``
    for a missing shared library, for example -- and is just as unusable.
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name == package:
        return f"{package} is not installed"
    return f"{package} failed to import ({type(exc).__name__}: {exc})"


try:
    import triton
    import triton.language as tl
except Exception as exc:  # Absent or broken install: launchers fail closed.
    triton = None
    tl = None
    _TRITON_UNAVAILABLE: str | None = _import_failure("triton", exc)
else:
    _TRITON_UNAVAILABLE = None


# Kernel definitions are guarded so that ``import silkern`` succeeds without a GPU
# stack; the launchers below raise LocalizationError if called anyway.
if triton is not None:

    @triton.jit
    def _rowwise_kernel(
        req_ids_ptr,
        block_table_ptr,
        tokens_ptr,
        out_ptr,
        counts_ptr,
        bt_stride0: tl.constexpr,
        bt_stride1: tl.constexpr,
        token_stride0: tl.constexpr,
        token_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        WIDTH: tl.constexpr,
        PADDED_WIDTH: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DCP_SIZE: tl.constexpr,
        DCP_RANK: tl.constexpr,
        DCP_INTERLEAVE: tl.constexpr,
        COMPACT_TO_FRONT: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_i64 = row.to(tl.int64)
        column = tl.arange(0, PADDED_WIDTH)
        in_width = column < WIDTH
        token = tl.load(
            tokens_ptr + row_i64 * token_stride0 + column * token_stride1,
            mask=in_width,
            other=-1,
        )
        valid_token = in_width & (token >= 0)
        owner = (token // DCP_INTERLEAVE) % DCP_SIZE
        owned = valid_token & (owner == DCP_RANK)
        local_token = (
            token // (DCP_SIZE * DCP_INTERLEAVE)
        ) * DCP_INTERLEAVE + token % DCP_INTERLEAVE
        logical_block = local_token // BLOCK_SIZE
        offset = local_token % BLOCK_SIZE
        in_table = owned & (logical_block >= 0) & (logical_block < BLOCK_TABLE_WIDTH)
        request = tl.load(req_ids_ptr + row)
        request_i64 = request.to(tl.int64)
        physical_block = tl.load(
            block_table_ptr
            + request_i64 * bt_stride0
            + logical_block * bt_stride1,
            mask=in_table,
            other=-1,
        )
        mapped_valid = in_table
        physical = physical_block * BLOCK_SIZE + offset
        valid_i = mapped_valid.to(tl.int32)
        total = tl.sum(valid_i, axis=0)

        if COMPACT_TO_FRONT:
            position = tl.cumsum(valid_i, axis=0) - valid_i
            tl.store(
                out_ptr + row_i64 * out_stride0 + column * out_stride1,
                tl.full((PADDED_WIDTH,), -1, tl.int32),
                mask=in_width,
            )
            tl.debug_barrier()
            tl.store(
                out_ptr + row_i64 * out_stride0 + position * out_stride1,
                physical,
                mask=mapped_valid,
            )
        else:
            value = tl.where(mapped_valid, physical, -1)
            tl.store(
                out_ptr + row_i64 * out_stride0 + column * out_stride1,
                value,
                mask=in_width,
            )
        tl.store(counts_ptr + row, total)

    @triton.jit
    def _map_tiles_kernel(
        req_ids_ptr,
        block_table_ptr,
        tokens_ptr,
        out_ptr,
        mapped_ptr,
        local_positions_ptr,
        tile_counts_ptr,
        bt_stride0: tl.constexpr,
        bt_stride1: tl.constexpr,
        token_stride0: tl.constexpr,
        token_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        mapped_stride0: tl.constexpr,
        mapped_stride1: tl.constexpr,
        positions_stride0: tl.constexpr,
        positions_stride1: tl.constexpr,
        tile_counts_stride0: tl.constexpr,
        tile_counts_stride1: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK_TABLE_WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DCP_SIZE: tl.constexpr,
        DCP_RANK: tl.constexpr,
        DCP_INTERLEAVE: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        DIRECT_OUTPUT: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        row_i64 = row.to(tl.int64)
        tile_i64 = tile.to(tl.int64)
        lane = tl.arange(0, TILE_SIZE)
        column = tile * TILE_SIZE + lane
        in_width = column < WIDTH
        token = tl.load(
            tokens_ptr + row_i64 * token_stride0 + column * token_stride1,
            mask=in_width,
            other=-1,
        )
        valid_token = in_width & (token >= 0)
        owner = (token // DCP_INTERLEAVE) % DCP_SIZE
        owned = valid_token & (owner == DCP_RANK)
        local_token = (
            token // (DCP_SIZE * DCP_INTERLEAVE)
        ) * DCP_INTERLEAVE + token % DCP_INTERLEAVE
        logical_block = local_token // BLOCK_SIZE
        offset = local_token % BLOCK_SIZE
        in_table = owned & (logical_block >= 0) & (logical_block < BLOCK_TABLE_WIDTH)
        request = tl.load(req_ids_ptr + row)
        request_i64 = request.to(tl.int64)
        physical_block = tl.load(
            block_table_ptr
            + request_i64 * bt_stride0
            + logical_block * bt_stride1,
            mask=in_table,
            other=-1,
        )
        physical = physical_block * BLOCK_SIZE + offset
        valid_i = in_table.to(tl.int32)
        local_position = tl.cumsum(valid_i, axis=0) - valid_i

        tl.store(
            mapped_ptr + row_i64 * mapped_stride0 + column * mapped_stride1,
            physical,
            mask=in_table,
        )
        tl.store(
            local_positions_ptr
            + row_i64 * positions_stride0
            + column * positions_stride1,
            tl.where(in_table, local_position, -1),
            mask=in_width,
        )
        tl.store(
            tile_counts_ptr
            + row_i64 * tile_counts_stride0
            + tile_i64 * tile_counts_stride1,
            tl.sum(valid_i, axis=0),
        )
        if DIRECT_OUTPUT:
            value = tl.where(in_table, physical, -1)
            tl.store(
                out_ptr + row_i64 * out_stride0 + column * out_stride1,
                value,
                mask=in_width,
            )

    @triton.jit
    def _tile_prefix_kernel(
        tile_counts_ptr,
        tile_offsets_ptr,
        counts_ptr,
        tile_counts_stride0: tl.constexpr,
        tile_counts_stride1: tl.constexpr,
        tile_offsets_stride0: tl.constexpr,
        tile_offsets_stride1: tl.constexpr,
        NUM_TILES: tl.constexpr,
        PADDED_TILES: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_i64 = row.to(tl.int64)
        tile = tl.arange(0, PADDED_TILES)
        in_tiles = tile < NUM_TILES
        tile_count = tl.load(
            tile_counts_ptr
            + row_i64 * tile_counts_stride0
            + tile * tile_counts_stride1,
            mask=in_tiles,
            other=0,
        )
        tile_offset = tl.cumsum(tile_count, axis=0) - tile_count
        tl.store(
            tile_offsets_ptr
            + row_i64 * tile_offsets_stride0
            + tile * tile_offsets_stride1,
            tile_offset,
            mask=in_tiles,
        )
        tl.store(counts_ptr + row, tl.sum(tile_count, axis=0))

    @triton.jit
    def _fill_output_kernel(
        out_ptr,
        elements,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offset, -1, mask=offset < elements)

    @triton.jit
    def _scatter_tiles_kernel(
        out_ptr,
        mapped_ptr,
        local_positions_ptr,
        tile_counts_ptr,
        tile_offsets_ptr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        mapped_stride0: tl.constexpr,
        mapped_stride1: tl.constexpr,
        positions_stride0: tl.constexpr,
        positions_stride1: tl.constexpr,
        tile_counts_stride0: tl.constexpr,
        tile_counts_stride1: tl.constexpr,
        tile_offsets_stride0: tl.constexpr,
        tile_offsets_stride1: tl.constexpr,
        WIDTH: tl.constexpr,
        TILE_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        row_i64 = row.to(tl.int64)
        tile_i64 = tile.to(tl.int64)
        lane = tl.arange(0, TILE_SIZE)
        column = tile * TILE_SIZE + lane
        in_width = column < WIDTH
        tile_count = tl.load(
            tile_counts_ptr
            + row_i64 * tile_counts_stride0
            + tile_i64 * tile_counts_stride1
        )
        tile_offset = tl.load(
            tile_offsets_ptr
            + row_i64 * tile_offsets_stride0
            + tile_i64 * tile_offsets_stride1
        )
        local_position = tl.load(
            local_positions_ptr
            + row_i64 * positions_stride0
            + column * positions_stride1,
            mask=in_width,
            other=-1,
        )
        mapped = tl.load(
            mapped_ptr + row_i64 * mapped_stride0 + column * mapped_stride1,
            mask=in_width,
            other=-1,
        )
        # Valid lanes are not necessarily the first ``tile_count`` input lanes.
        valid = in_width & (local_position >= 0) & (local_position < tile_count)
        destination = tile_offset + local_position
        tl.store(
            out_ptr + row_i64 * out_stride0 + destination * out_stride1,
            mapped,
            mask=valid,
        )


def _validate_launch_config(
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int,
    compact_valid_to_front: bool,
    num_warps: int,
) -> tuple[int, int, int, int, int]:
    """Validate and normalize host scalars before Triton specializes a kernel."""
    _validate_dcp_config(dcp_size, dcp_rank, dcp_interleave)
    _validate_compact_flag(compact_valid_to_front)
    if not isinstance(block_size, Integral) or isinstance(block_size, bool):
        raise LocalizationError("block_size must be a positive integer divisible by dcp_interleave")
    # Integral includes NumPy scalar integers. Normalize before multiplication:
    # fixed-width host arithmetic could otherwise wrap around the bounds check.
    block_size, dcp_size, dcp_rank, dcp_interleave = (
        int(value) for value in (block_size, dcp_size, dcp_rank, dcp_interleave)
    )
    if block_size < 1 or block_size % dcp_interleave:
        raise LocalizationError("block_size must be a positive integer divisible by dcp_interleave")
    if max(block_size, dcp_size, dcp_rank, dcp_interleave,
           dcp_size * dcp_interleave) > 2**31 - 1:
        raise LocalizationError("CUDA geometry and dcp_size * dcp_interleave must fit int32")
    if (
        not isinstance(num_warps, Integral)
        or isinstance(num_warps, bool)
        or num_warps not in (4, 8)
    ):
        raise LocalizationError("num_warps must be 4 or 8")
    return block_size, dcp_size, dcp_rank, dcp_interleave, int(num_warps)


def _validate_disjoint_storage(tensors, *, read_only_count: int) -> None:
    """Reject written buffers that share storage or overlap memory, without device work.

    Call only after validating contiguous, nonempty buffers on one device. The
    first ``read_only_count`` tensors are inputs, which the kernels only read, so
    they may alias one another. Every later tensor is written and needs storage
    of its own: it may not share a storage object with any other buffer, even as
    a non-overlapping view of one tensor, nor overlap one in memory. DLPack
    imports can wrap overlapping slices in separate storage objects with
    different base pointers, so storage identity alone is insufficient.
    """
    regions = []
    for tensor in tensors:
        start = tensor.data_ptr()
        regions.append((tensor.untyped_storage().data_ptr(), start,
                        start + tensor.numel() * tensor.element_size()))
    for index in range(read_only_count, len(regions)):
        storage, start, end = regions[index]
        for other_storage, other_start, other_end in regions[:index]:
            if storage == other_storage or (start < other_end and other_start < end):
                raise LocalizationError(
                    "written localization buffers must use distinct storage and "
                    "nonoverlapping memory"
                )


def localize_rowwise(
    req_ids,
    block_table,
    tokens,
    out,
    counts,
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int = 1,
    compact_valid_to_front: bool = True,
    num_warps: int = 8,
) -> None:
    """Row-wide stable localization: one program per selection row, one launch.

    Every buffer is a preallocated contiguous int32 CUDA tensor supplied by the
    caller; nothing is allocated here, so the launch is safe to capture in a
    CUDA graph and replay at fixed addresses. ``out`` must be shaped like
    ``tokens`` and ``counts`` must hold one element per row. The three inputs
    are only read and may share storage with one another; ``out`` and
    ``counts`` each need storage of their own, shared with no other buffer even
    as a non-overlapping view, and must not overlap any buffer in memory.
    Launches use the current stream on the tensors' device and restore the
    caller's current device afterward. Lazy negation views are rejected: their
    logical values differ from their raw storage.

    The row width is capped at :data:`silkern.contract.MAX_ROW_WIDTH` because the
    row-wide scan is a single ``cumsum`` over a power-of-two-padded row. The
    hierarchical implementation currently enforces the same qualified limit.

    ``req_ids`` values are a caller precondition. They live on the device, so the
    launcher cannot bound-check them without a host synchronization -- which a
    capture-safe launcher must never do -- and an id outside ``block_table``
    reads past the table rather than raising. See ``docs/contract.md``.

    Raises :class:`~silkern.errors.LocalizationError` on any unsupported geometry
    or buffer binding rather than falling back to a different code path.
    """
    if triton is None:
        raise LocalizationError(
            f"triton is required for stable localization: {_TRITON_UNAVAILABLE}"
        )
    import torch

    tensors = (req_ids, block_table, tokens, out, counts)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise LocalizationError("all localization buffers must be torch tensors")
    if any(not tensor.is_cuda for tensor in tensors):
        raise LocalizationError("all localization buffers must be CUDA tensors")
    if any(tensor.device != tokens.device for tensor in tensors):
        raise LocalizationError("all localization buffers must be on the same device")
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise LocalizationError("all localization buffers must use int32")
    if any(tensor.is_neg() for tensor in tensors):
        raise LocalizationError("localization buffers must not use lazy negation views")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise LocalizationError("stable localization requires contiguous tensors")
    if tokens.ndim != 2 or out.shape != tokens.shape:
        raise LocalizationError("tokens and out must have the same 2D shape")
    if block_table.ndim != 2:
        raise LocalizationError("block_table must be 2D")
    batch, width = tokens.shape
    if req_ids.shape != (batch,) or counts.shape != (batch,):
        raise LocalizationError("req_ids and counts must contain one element per row")
    if (
        batch < 1
        or width < 1
        or block_table.shape[0] < 1
        or block_table.shape[1] < 1
    ):
        raise LocalizationError(
            "batch, width, and both block-table dimensions must be positive"
        )
    if width > MAX_ROW_WIDTH:
        raise LocalizationError(
            f"row width {width} exceeds the current {MAX_ROW_WIDTH}-element limit"
        )
    if batch > 2**31 - 1:
        raise LocalizationError("batch must fit int32 for the CUDA launch grid")
    _validate_disjoint_storage(tensors, read_only_count=3)
    block_size, dcp_size, dcp_rank, dcp_interleave, num_warps = _validate_launch_config(
        block_size, dcp_size, dcp_rank, dcp_interleave, compact_valid_to_front, num_warps
    )

    # Triton chooses its device and stream from the current CUDA context.
    # Tensor pointer validation alone does not select their device.
    with torch.cuda.device(tokens.device):
        padded = triton.next_power_of_2(width)
        compact = compact_valid_to_front and dcp_size > 1
        _rowwise_kernel[(batch,)](
            req_ids,
            block_table,
            tokens,
            out,
            counts,
            block_table.stride(0),
            block_table.stride(1),
            tokens.stride(0),
            tokens.stride(1),
            out.stride(0),
            out.stride(1),
            WIDTH=width,
            PADDED_WIDTH=padded,
            BLOCK_TABLE_WIDTH=block_table.shape[1],
            BLOCK_SIZE=block_size,
            DCP_SIZE=dcp_size,
            DCP_RANK=dcp_rank,
            DCP_INTERLEAVE=dcp_interleave,
            COMPACT_TO_FRONT=compact,
            num_warps=num_warps,
        )


def localize_hierarchical(
    req_ids,
    block_table,
    tokens,
    out,
    counts,
    mapped_workspace,
    local_positions_workspace,
    tile_counts_workspace,
    tile_offsets_workspace,
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int = 1,
    compact_valid_to_front: bool = True,
    tile_size: int = DEFAULT_TILE_SIZE,
    num_warps: int = 4,
) -> None:
    """Hierarchical stable localization: bounded tile scans plus a stable scatter.

    ``mapped_workspace`` and ``local_positions_workspace`` must match the input
    shape.  The two tile workspaces must have shape
    ``(batch, ceil(width / tile_size))``.  Every buffer is an int32 contiguous
    CUDA tensor on one device. The three inputs are only read and may share
    storage with one another; each output and workspace needs storage of its
    own, shared with no other buffer even as a non-overlapping view, and must
    not overlap any buffer in memory, even through separately imported storage
    objects. All stages use the current stream on that device and restore the
    caller's current device afterward. Lazy negation views are rejected.

    The compacting path launches four deterministic stages: per-tile mapping and
    local prefix, per-row tile prefix, output initialization, and stable scatter.
    The non-compacting path writes columns directly in the mapping stage and
    launches only the tile-prefix stage to produce counts.

    ``req_ids`` values are a caller precondition. They live on the device, so the
    launcher cannot bound-check them without a host synchronization -- which a
    capture-safe launcher must never do -- and an id outside ``block_table``
    reads past the table rather than raising. See ``docs/contract.md``.
    """
    if triton is None:
        raise LocalizationError(
            f"triton is required for stable localization: {_TRITON_UNAVAILABLE}"
        )
    import torch

    tensors = (
        req_ids,
        block_table,
        tokens,
        out,
        counts,
        mapped_workspace,
        local_positions_workspace,
        tile_counts_workspace,
        tile_offsets_workspace,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must be torch tensors"
        )
    if any(not tensor.is_cuda for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must be CUDA tensors"
        )
    if any(tensor.device != tokens.device for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must be on the same device"
        )
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise LocalizationError(
            "all hierarchical localization buffers must use int32"
        )
    if any(tensor.is_neg() for tensor in tensors):
        raise LocalizationError("localization buffers must not use lazy negation views")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise LocalizationError(
            "hierarchical stable localization requires contiguous tensors"
        )
    if tokens.ndim != 2 or out.shape != tokens.shape:
        raise LocalizationError("tokens and out must have the same 2D shape")
    if block_table.ndim != 2:
        raise LocalizationError("block_table must be 2D")
    batch, width = tokens.shape
    if req_ids.shape != (batch,) or counts.shape != (batch,):
        raise LocalizationError("req_ids and counts must contain one element per row")
    if (
        batch < 1
        or width < 1
        or block_table.shape[0] < 1
        or block_table.shape[1] < 1
    ):
        raise LocalizationError(
            "batch, width, and both block-table dimensions must be positive"
        )

    shapes = workspace_shapes(
        batch,
        width,
        tile_size=tile_size,
    )
    observed_shapes = {
        "mapped": tuple(mapped_workspace.shape),
        "local_positions": tuple(local_positions_workspace.shape),
        "tile_counts": tuple(tile_counts_workspace.shape),
        "tile_offsets": tuple(tile_offsets_workspace.shape),
    }
    for name, expected in shapes.items():
        if observed_shapes[name] != expected:
            raise LocalizationError(
                f"{name} workspace shape mismatch: expected {expected}, "
                f"observed {observed_shapes[name]}"
            )

    _validate_disjoint_storage(tensors, read_only_count=3)
    block_size, dcp_size, dcp_rank, dcp_interleave, num_warps = _validate_launch_config(
        block_size, dcp_size, dcp_rank, dcp_interleave, compact_valid_to_front, num_warps
    )

    # workspace_shapes validates Integral values but its normalization is local.
    # Triton constexpr arithmetic requires the corresponding Python int.
    tile_size = int(tile_size)
    with torch.cuda.device(tokens.device):
        num_tiles = shapes["tile_counts"][1]
        compact = compact_valid_to_front and dcp_size > 1
        _map_tiles_kernel[(batch, num_tiles)](
            req_ids,
            block_table,
            tokens,
            out,
            mapped_workspace,
            local_positions_workspace,
            tile_counts_workspace,
            block_table.stride(0),
            block_table.stride(1),
            tokens.stride(0),
            tokens.stride(1),
            out.stride(0),
            out.stride(1),
            mapped_workspace.stride(0),
            mapped_workspace.stride(1),
            local_positions_workspace.stride(0),
            local_positions_workspace.stride(1),
            tile_counts_workspace.stride(0),
            tile_counts_workspace.stride(1),
            WIDTH=width,
            BLOCK_TABLE_WIDTH=block_table.shape[1],
            BLOCK_SIZE=block_size,
            DCP_SIZE=dcp_size,
            DCP_RANK=dcp_rank,
            DCP_INTERLEAVE=dcp_interleave,
            TILE_SIZE=tile_size,
            DIRECT_OUTPUT=not compact,
            num_warps=num_warps,
        )
        padded_tiles = triton.next_power_of_2(num_tiles)
        _tile_prefix_kernel[(batch,)](
            tile_counts_workspace,
            tile_offsets_workspace,
            counts,
            tile_counts_workspace.stride(0),
            tile_counts_workspace.stride(1),
            tile_offsets_workspace.stride(0),
            tile_offsets_workspace.stride(1),
            NUM_TILES=num_tiles,
            PADDED_TILES=padded_tiles,
            num_warps=1,
        )
        if compact:
            elements = batch * width
            fill_block = 256
            _fill_output_kernel[(triton.cdiv(elements, fill_block),)](
                out,
                elements,
                BLOCK=fill_block,
                num_warps=4,
            )
            _scatter_tiles_kernel[(batch, num_tiles)](
                out,
                mapped_workspace,
                local_positions_workspace,
                tile_counts_workspace,
                tile_offsets_workspace,
                out.stride(0),
                out.stride(1),
                mapped_workspace.stride(0),
                mapped_workspace.stride(1),
                local_positions_workspace.stride(0),
                local_positions_workspace.stride(1),
                tile_counts_workspace.stride(0),
                tile_counts_workspace.stride(1),
                tile_offsets_workspace.stride(0),
                tile_offsets_workspace.stride(1),
                WIDTH=width,
                TILE_SIZE=tile_size,
                num_warps=num_warps,
            )
