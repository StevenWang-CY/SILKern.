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
    except Exception as exc:  # A broken installation can fail with any error type.
        raise LocalizationError(
            "MLX is unavailable; on Apple silicon with macOS 14 or later, "
            "install silkern[mlx] using native arm64 Python "
            f"({type(exc).__name__}: {exc})"
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
# Array dimensions are read at run time: one compiled kernel serves every batch
# and block-table shape, and every row width up to 256 * ITEMS.
_METAL_SOURCE = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint simd_lane = lane % 32;
    const uint simd_group = lane / 32;
    const int width = tokens_shape[1];
    const int requests = table_shape[0];
    const int table_width = table_shape[1];
    const int request = req[row];
    const bool request_valid = request >= 0 && request < requests;
    int values[ITEMS];
    bool valid[ITEMS];
    int local_count = 0;
    for (uint i = 0; i < ITEMS; ++i) {
        const uint col = lane * ITEMS + i;
        bool keep = false;
        int physical = -1;
        if (col < width) {
            const int token = tokens[row * width + col];
            if (request_valid && token >= 0 && (token / INTERLEAVE) % DCP == RANK) {
                const int local = (token / (DCP * INTERLEAVE)) * INTERLEAVE
                                  + token % INTERLEAVE;
                const int block = local / BLOCK;
                if (block < table_width) {
                    const int page = table[uint(request) * table_width + uint(block)];
                    // Unsigned arithmetic avoids C++ signed-overflow UB. The
                    // public value contract requires representable int32 slots.
                    physical = as_type<int>(uint(page) * uint(BLOCK) + uint(local % BLOCK));
                    keep = true;
                }
            }
            out[row * width + col] = COMPACT ? -1 : physical;
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
            if (valid[i]) out[row * width + uint(position++)] = values[i];
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


def _divmod_nonnegative(mx, s, values, divisor):
    """Exact ``divmod`` of int32 values in ``[0, 2**31)`` by a positive int, on ``s``.

    MLX 0.32.3's integer floor_divide creates part of its sign correction on
    the default stream whatever stream it is given, and repeated chains of
    multi-output divmod fail inside mx.compile. Division by multiplication is
    elementwise: with l = ceil(log2(d)), n // d == (n * ceil(2**(31 + l) / d))
    >> (31 + l), and the product stays below 2**63 (Granlund and Montgomery,
    PLDI 1994, Theorem 4.2).
    """
    if divisor == 1:  # The default interleave: skip six operations.
        return values, mx.zeros_like(values, stream=s)
    shift = 31 + (divisor - 1).bit_length()
    scaled = mx.multiply(mx.astype(values, mx.int64, stream=s), -(-(1 << shift) // divisor),
                         stream=s)
    quotient = mx.astype(mx.right_shift(scaled, shift, stream=s), mx.int32, stream=s)
    return quotient, mx.subtract(values, mx.multiply(quotient, divisor, stream=s), stream=s)


def _localize_native(mx, s, req, table, tokens, block, dcp, rank, interleave, compact):
    """Linear-work MLX baseline; scatter destinations are a unique permutation.

    Every operation names the stream ``s``. Python operators and ``mx.stream``
    would follow MLX's default device, which all threads share.
    """
    requests, table_width = table.shape
    # One unsigned comparison rejects negative and too-large request ids. MLX
    # cannot infer shapes through this View, so mx.compile(..., shapeless=True)
    # raises on its first call instead of replaying the shapes read above.
    request_valid = mx.less(mx.view(req, mx.uint32, stream=s), requests, stream=s)
    safe_tokens = mx.maximum(tokens, 0, stream=s)
    group, lane = _divmod_nonnegative(mx, s, safe_tokens, interleave)
    period, owner = _divmod_nonnegative(mx, s, group, dcp)
    local = mx.add(mx.multiply(period, interleave, stream=s), lane, stream=s)
    logical_block, offset = _divmod_nonnegative(mx, s, local, block)
    valid = mx.logical_and(
        mx.logical_and(mx.greater_equal(tokens, 0, stream=s), mx.equal(owner, rank, stream=s),
                       stream=s),
        mx.logical_and(mx.less(logical_block, table_width, stream=s),
                       mx.expand_dims(request_valid, 1, stream=s), stream=s),
        stream=s,
    )
    # Clamp before the gather, including invalid requests. Masking a result
    # after an out-of-bounds gather would not make that access safe.
    row_start = mx.multiply(mx.clip(req, 0, requests - 1, stream=s), table_width, stream=s)
    addresses = mx.add(mx.expand_dims(row_start, 1, stream=s),
                       mx.minimum(logical_block, table_width - 1, stream=s), stream=s)
    pages = mx.take(mx.reshape(table, (-1,), stream=s), addresses, stream=s)
    slots = mx.add(mx.multiply(pages, block, stream=s), offset, stream=s)
    mapped = mx.where(valid, slots, -1, stream=s)
    valid_i = mx.astype(valid, mx.int32, stream=s)
    counts = mx.astype(mx.sum(valid_i, axis=1, stream=s), mx.int32, stream=s)
    if not compact:
        return mapped, counts
    prefix = mx.subtract(mx.cumsum(valid_i, axis=1, stream=s), valid_i, stream=s)
    columns = mx.expand_dims(mx.arange(tokens.shape[1], dtype=mx.int32, stream=s), 0, stream=s)
    # Invalid entries go to unique tail slots, never to a valid destination.
    tail = mx.subtract(mx.add(mx.expand_dims(counts, 1, stream=s), columns, stream=s), prefix,
                       stream=s)
    destinations = mx.where(valid, prefix, tail, stream=s)
    empty = mx.full(tokens.shape, -1, dtype=mx.int32, stream=s)
    return mx.put_along_axis(empty, destinations, mapped, axis=1, stream=s), counts


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
    Every operation is created on the selected stream explicitly; the caller's
    default device and streams are never changed, even temporarily.
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
    ``mx.compile`` retraces, and so revalidates, for every new input shape;
    ``mx.compile(..., shapeless=True)`` is refused on its first call.
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
        raise LocalizationError(
            "stream must be an MLX Stream, ThreadLocalStream, Device, or DeviceType"
        )
    device = stream.device if isinstance(stream, stream_types) else stream
    if device is None:
        device = mx.gpu if backend == "metal" else mx.default_device()
    metal_available = mx.metal.is_available()
    use_metal = backend == "metal" or (backend == "auto" and device == mx.gpu and metal_available)
    if use_metal and (not metal_available or device != mx.gpu):
        raise LocalizationError("the Metal backend requires an available Apple GPU and GPU stream")
    # MLX's default device is process-wide: selecting it with ``mx.stream`` here
    # would race with other threads, so each operation receives this stream.
    selected_stream = stream if isinstance(stream, stream_types) else mx.default_stream(device)
    compact = compact_valid_to_front and dcp_size > 1
    if not use_metal:
        return _localize_native(mx, selected_stream, req_ids, block_table, token_indices,
                                block_size, dcp_size, dcp_rank, dcp_interleave, compact)
    batch, width = token_indices.shape
    out, counts = _metal_kernel()(
        inputs=[req_ids, block_table, token_indices],
        # Only the per-thread chunk length and the scalar geometry specialize
        # the kernel; batch, width, and table shape are read at run time.
        template=[("ITEMS", (width + 255) // 256), ("BLOCK", block_size),
                  ("DCP", dcp_size), ("RANK", dcp_rank), ("INTERLEAVE", dcp_interleave),
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
