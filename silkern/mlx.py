"""Functional MLX localization, with a fused stable-compaction Metal kernel.

Optional dependencies are imported only when called. Outputs are new lazy MLX
arrays: the CUDA launchers' allocation-free and graph-replay guarantees do not
apply here. See ``docs/apple-mlx.md`` for the complete platform contract.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import TYPE_CHECKING, Any

from silkern.contract import MAX_ROW_WIDTH, _validate_compact_flag, _validate_dcp_config
from silkern.errors import LocalizationError

if TYPE_CHECKING:
    import mlx.core as mx

_INT32_MAX = 2**31 - 1


def _load_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise LocalizationError(
            "MLX is unavailable; on Apple silicon with macOS 14 or later, "
            "install silkern[mlx] using native arm64 Python"
        ) from exc
    return mx


def _validate_inputs(mx, req_ids, block_table, token_indices, block_size,
                     dcp_size, dcp_rank, dcp_interleave, compact_valid_to_front):
    _validate_dcp_config(dcp_size, dcp_rank, dcp_interleave)
    _validate_compact_flag(compact_valid_to_front)
    if not isinstance(block_size, Integral) or isinstance(block_size, bool) or block_size < 1:
        raise LocalizationError("block_size must be a positive integer")
    # Integral includes fixed-width NumPy scalars. Normalize before products so
    # host-side validation itself cannot overflow and accept unsafe geometry.
    block_size, dcp_size, dcp_rank, dcp_interleave = map(
        int, (block_size, dcp_size, dcp_rank, dcp_interleave)
    )
    if block_size % dcp_interleave:
        raise LocalizationError("block_size must be divisible by dcp_interleave")
    if max(block_size, dcp_size, dcp_rank, dcp_interleave,
           dcp_size * dcp_interleave) > _INT32_MAX:
        raise LocalizationError("MLX geometry and dcp_size * dcp_interleave must fit int32")
    for name, value, ndim in (("req_ids", req_ids, 1), ("block_table", block_table, 2),
                              ("token_indices", token_indices, 2)):
        if not isinstance(value, mx.array):
            raise LocalizationError(f"{name} must be an mlx.core.array")
        if value.dtype != mx.int32:
            raise LocalizationError(f"{name} must have dtype int32")
        if value.ndim != ndim or any(size < 1 for size in value.shape):
            raise LocalizationError(f"{name} must be a nonempty {ndim}-dimensional array")
        if value.size > _INT32_MAX:
            raise LocalizationError(f"{name} must contain at most {_INT32_MAX} elements")
    if req_ids.shape[0] != token_indices.shape[0]:
        raise LocalizationError("req_ids must contain one request id per row")
    if token_indices.shape[1] > MAX_ROW_WIDTH:
        raise LocalizationError(f"row width exceeds the current {MAX_ROW_WIDTH}-element limit")
    return block_size, dcp_size, dcp_rank, dcp_interleave


# A thread processes a contiguous chunk of the selection row. SIMD scans count
# the survivors before each chunk; eight shared totals connect the SIMD groups.
# No atomic reservation is involved. The barrier separates tail initialization
# from scatter, including writes by other SIMD groups in this threadgroup.
_METAL_SOURCE = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint simd_lane = lane % 32;
    const uint simd_group = lane / 32;
    const int request = req[row];
    const bool request_valid = request >= 0 && request < REQUESTS;
    int values[ITEMS];
    bool valid[ITEMS];
    int local_count = 0;
    for (uint i = 0; i < ITEMS; ++i) {
        const uint col = lane * ITEMS + i;
        bool keep = false;
        int physical = -1;
        if (col < WIDTH) {
            const int token = tokens[row * WIDTH + col];
            if (request_valid && token >= 0 && (token / INTERLEAVE) % DCP == RANK) {
                const int local = (token / (DCP * INTERLEAVE)) * INTERLEAVE
                                  + token % INTERLEAVE;
                const int block = local / BLOCK;
                if (block < TABLE_WIDTH) {
                    const int page = table[uint(request) * TABLE_WIDTH + uint(block)];
                    // Unsigned arithmetic avoids C++ signed-overflow UB. The
                    // public value contract requires representable int32 slots.
                    physical = as_type<int>(uint(page) * uint(BLOCK) + uint(local % BLOCK));
                    keep = true;
                }
            }
            out[row * WIDTH + col] = COMPACT ? -1 : physical;
        }
        values[i] = physical;
        valid[i] = keep;
        local_count += int(keep);
    }
    const int lane_prefix = simd_prefix_exclusive_sum(local_count);
    const int simd_total = simd_sum(local_count);
    threadgroup int totals[8];
    if (simd_lane == 0) totals[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    int preceding = 0;
    int total = 0;
    for (uint g = 0; g < 8; ++g) {
        if (g < simd_group) preceding += totals[g];
        total += totals[g];
    }
    if (lane == 0) counts[row] = total;
    if (COMPACT) {
        int position = preceding + lane_prefix;
        for (uint i = 0; i < ITEMS; ++i) {
            if (valid[i]) out[row * WIDTH + uint(position++)] = values[i];
        }
    }
"""


@lru_cache(maxsize=1)
def _metal_kernel():
    mx = _load_mlx()
    return mx.fast.metal_kernel(
        name="silkern_localize_int32",
        input_names=["req", "table", "tokens"],
        output_names=["out", "counts"],
        source=_METAL_SOURCE,
        ensure_row_contiguous=True,
    )


def _localize_native(mx, req, table, tokens, block, dcp, rank, interleave, compact):
    """Linear-work MLX baseline; scatter destinations are a unique permutation."""
    safe_tokens = mx.maximum(tokens, 0)
    local = (safe_tokens // (dcp * interleave)) * interleave + safe_tokens % interleave
    logical_block = local // block
    request_valid = (req >= 0) & (req < table.shape[0])
    valid = ((tokens >= 0) & ((safe_tokens // interleave) % dcp == rank)
             & (logical_block < table.shape[1]) & request_valid[:, None])
    # Clamp before the gather, including invalid requests. Masking a result
    # after an out-of-bounds gather would not make that access safe.
    addresses = (mx.clip(req, 0, table.shape[0] - 1)[:, None] * table.shape[1]
                 + mx.minimum(logical_block, table.shape[1] - 1))
    pages = mx.take(mx.reshape(table, (-1,)), addresses)
    mapped = mx.where(valid, pages * block + local % block, -1)
    valid_i = valid.astype(mx.int32)
    counts = mx.sum(valid_i, axis=1).astype(mx.int32)
    if not compact:
        return mapped, counts
    prefix = mx.cumsum(valid_i, axis=1) - valid_i
    columns = mx.arange(tokens.shape[1], dtype=mx.int32)[None, :]
    # Invalid entries go to unique tail slots, never to a valid destination.
    destinations = mx.where(valid, prefix, counts[:, None] + columns - prefix)
    out = mx.put_along_axis(mx.full(tokens.shape, -1, dtype=mx.int32),
                            destinations, mapped, axis=1)
    return out, counts


def localize_mlx(
    req_ids: mx.array,
    block_table: mx.array,
    token_indices: mx.array,
    *,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_interleave: int = 1,
    compact_valid_to_front: bool = True,
    backend: str = "auto",
    stream: mx.Stream | mx.ThreadLocalStream | mx.Device | mx.DeviceType | None = None,
) -> tuple[mx.array, mx.array]:
    """Return lazy ``(out, counts)`` int32 arrays with stable localization.

    ``auto`` uses Metal on an Apple GPU stream, otherwise native MLX operations.
    ``metal`` requires Metal and a GPU stream; ``mlx`` uses native operations.
    An explicit stream is respected, and the caller's default stream is restored.
    ``mx.ThreadLocalStream`` resolves to the calling thread's own stream.
    With ``backend='metal'`` and no stream, the Apple GPU is selected explicitly.

    Shapes, scalar geometry, and int32 dtypes are checked without synchronization.
    Inputs may be strided; Metal makes contiguous copies as needed. Request ids
    outside the table produce an all-invalid row with count zero. Negative page
    entries remain valid and are translated verbatim, as in the Python oracle.
    Physical slots must fit signed int32; checking device values is the caller's
    responsibility. DCP-1 bypasses compaction, matching the shared contract.

    Evaluate with ``mx.eval(out, counts)`` before observing or timing results.
    This API allocates outputs and may allocate intermediate/copy buffers.
    """
    if backend not in ("auto", "metal", "mlx"):
        raise LocalizationError("backend must be 'auto', 'metal', or 'mlx'")
    mx = _load_mlx()
    block_size, dcp_size, dcp_rank, dcp_interleave = _validate_inputs(
        mx, req_ids, block_table, token_indices, block_size, dcp_size,
        dcp_rank, dcp_interleave, compact_valid_to_front
    )
    stream_types = (mx.Stream, mx.ThreadLocalStream)
    if stream is not None and not isinstance(stream, (*stream_types, mx.Device, mx.DeviceType)):
        raise LocalizationError("stream must be an MLX Stream, ThreadLocalStream, or Device")
    device = stream.device if isinstance(stream, stream_types) else stream
    if device is None:
        device = mx.gpu if backend == "metal" else mx.default_device()
    metal_available = mx.metal.is_available()
    use_metal = backend == "metal" or (backend == "auto" and device == mx.gpu and metal_available)
    if use_metal and (not metal_available or device != mx.gpu):
        raise LocalizationError("the Metal backend requires an available Apple GPU and GPU stream")
    selected_stream = stream if isinstance(stream, stream_types) else mx.default_stream(device)
    compact = compact_valid_to_front and dcp_size > 1
    with mx.stream(selected_stream):
        if not use_metal:
            return _localize_native(mx, req_ids, block_table, token_indices, block_size,
                                    dcp_size, dcp_rank, dcp_interleave, compact)
        batch, width = token_indices.shape
        out, counts = _metal_kernel()(
            inputs=[req_ids, block_table, token_indices],
            template=[("WIDTH", width), ("ITEMS", (width + 255) // 256),
                      ("REQUESTS", block_table.shape[0]), ("TABLE_WIDTH", block_table.shape[1]),
                      ("BLOCK", block_size), ("DCP", dcp_size),
                      ("RANK", dcp_rank), ("INTERLEAVE", dcp_interleave),
                      ("COMPACT", compact)],
            # MLX dispatch dimensions are signed 32-bit. Put rows on a
            # separate axis so a legal batch cannot overflow batch * 256.
            grid=(256, batch, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, width), (batch,)],
            output_dtypes=[mx.int32, mx.int32],
            stream=selected_stream,
        )
        return out, counts
