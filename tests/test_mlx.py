"""Oracle-backed MLX regressions on CPU and Apple Metal; no CUDA execution."""

from __future__ import annotations

import random

import pytest

from silkern import LocalizationError, localize_mlx, localize_reference

pytestmark = pytest.mark.mlx
mx = pytest.importorskip("mlx.core")


@pytest.fixture(params=[
    "mlx_cpu",
    pytest.param("mlx_gpu", marks=pytest.mark.metal),
    pytest.param("metal", marks=pytest.mark.metal),
])
def execution(request):
    if request.param != "mlx_cpu" and not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    device = mx.cpu if request.param == "mlx_cpu" else mx.gpu
    # Input-generation operations must follow the requested test device too.
    with mx.stream(device):
        yield "metal" if request.param == "metal" else "mlx", device


def _check(req, table, tokens, execution, **config):
    backend, stream = execution
    arrays = [mx.array(value, dtype=mx.int32) for value in (req, table, tokens)]
    out, counts = localize_mlx(*arrays, backend=backend, stream=stream, **config)
    assert (out.tolist(), counts.tolist()) == localize_reference(req, table, tokens, **config)
    assert [a.tolist() for a in arrays] == [req, table, tokens]
    assert out.dtype == counts.dtype == mx.int32
    return out, counts


@pytest.mark.parametrize("width", [1, 2, 31, 32, 33, 127, 255, 256, 257, 513, 2048, 4095, 4096])
def test_scan_boundaries_and_sparse_survivors(width, execution):
    rng = random.Random(width)
    req = [2, 0, 2, 1]
    table = [[rng.randrange(-2, 128) for _ in range(17)] for _ in range(3)]
    rows = [[rng.randrange(-32, 1800) for _ in range(width)] for _ in req]
    _check(req, table, rows, execution, block_size=16, dcp_size=4, dcp_rank=3,
           dcp_interleave=4)


@pytest.mark.parametrize("dcp", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4, 8])
@pytest.mark.parametrize("compact", [True, False])
def test_geometry_and_compaction(execution, dcp, interleave, compact):
    _check([1, 0], [[-1, 3], [7, -2]],
           [[63, -1, 0, 1, 4, 8, 7, 128, 7], [0, 5, 2, 99, 1, -3, 14, 15, 127]],
           execution, block_size=8, dcp_size=dcp, dcp_rank=dcp - 1,
           dcp_interleave=interleave, compact_valid_to_front=compact)


def test_valid_negative_page_and_minus_one_not_sentinel_for_count(execution):
    out, counts = _check([0], [[-1]], [[3, -1, 2, 3]], execution,
                         block_size=2, dcp_size=2, dcp_rank=1)
    assert out.tolist() == [[-1, -1, -1, -1]]
    assert counts.tolist() == [2]


@pytest.mark.parametrize("fill", [-1, 0, 2**31 - 1])
def test_all_invalid_or_all_owned(execution, fill):
    _check([0, 0], [[1]], [[fill] * 4096] * 2, execution,
           block_size=64, dcp_size=2, dcp_rank=0)


def test_invalid_requests_are_masked_before_gather(execution):
    backend, stream = execution
    out, counts = localize_mlx(mx.array([-1, 1, -(2**31), 2**31 - 1], dtype=mx.int32),
                              mx.array([[0]], dtype=mx.int32),
                              mx.zeros((4, 3), dtype=mx.int32),
                              block_size=1, dcp_size=2, dcp_rank=0,
                              backend=backend, stream=stream)
    assert out.tolist() == [[-1] * 3] * 4
    assert counts.tolist() == [0] * 4


def test_strided_inputs(execution):
    backend, stream = execution
    req = mx.array([0, 99, 1, 99], dtype=mx.int32)[::2]
    table = mx.array([[2, 7], [4, -1], [3, 8]], dtype=mx.int32).T
    tokens = mx.arange(60, dtype=mx.int32).reshape(10, 6).T[::3, ::2]
    config = dict(block_size=8, dcp_size=2, dcp_rank=0)
    out, counts = localize_mlx(req, table, tokens, backend=backend, stream=stream, **config)
    assert (out.tolist(), counts.tolist()) == localize_reference(
        req.tolist(), table.tolist(), tokens.tolist(), **config)


def test_int32_limits_without_intermediate_overflow(execution):
    _check([0], [[1, -1]], [[2**31 - 1, 2**31 - 2, -(2**31), 0]], execution,
           block_size=2**30, dcp_size=2, dcp_rank=1)
    _check([0], [[2**31 - 1, -(2**31)]], [[0, 2]], execution,
           block_size=1, dcp_size=2, dcp_rank=0)


@pytest.mark.parametrize("block,dcp,interleave,width", [
    (9, 3, 3, 37), (35, 5, 5, 259), (17, 7, 1, 1025),
    (2147483646, 1, 3, 33), (715827882, 3, 3, 257),
    (3, 715827882, 3, 129),
])
def test_non_power_of_two_divisors_and_large_indices(execution, block, dcp, interleave, width):
    rng = random.Random(block + dcp + width)
    # Keep translated values representable even with very large page sizes.
    tables = [[0, -1], [-1, 0]]
    limit = min(2**31 - 1, block * dcp * 2)
    specials = [-(2**31), -1, 0, 1, 2**31 - 1, 2**31 - 2, limit - 1, limit]
    rows = [specials + [rng.randrange(limit) for _ in range(width - len(specials))]
            for _ in tables]
    _check([1, 0], tables, rows, execution, block_size=block, dcp_size=dcp,
           dcp_rank=dcp - 1, dcp_interleave=interleave)


def test_broadcast_and_reversed_inputs(execution):
    backend, stream = execution
    req = mx.broadcast_to(mx.array([0], dtype=mx.int32), (3,))
    table = mx.array([[8, 3, -1, 2]], dtype=mx.int32)[:, ::-1]
    tokens = mx.broadcast_to(mx.arange(67, dtype=mx.int32)[::-2][None, :], (3, 34))
    options = dict(block_size=4, dcp_size=3, dcp_rank=2)
    out, counts = localize_mlx(req, table, tokens, backend=backend, stream=stream, **options)
    assert (out.tolist(), counts.tolist()) == localize_reference(
        req.tolist(), table.tolist(), tokens.tolist(), **options)


@pytest.mark.parametrize("config", [
    {"block_size": True}, {"block_size": 0}, {"block_size": 2.5},
    {"dcp_size": 0}, {"dcp_size": True}, {"dcp_rank": 2},
    {"dcp_interleave": 3}, {"dcp_interleave": 0},
    {"compact_valid_to_front": 1}, {"block_size": 2**31},
    {"dcp_size": 2**30, "dcp_interleave": 2},
    {"backend": "cuda"}, {"stream": "cpu"},
])
def test_metadata_validation(config):
    options = dict(block_size=4, dcp_size=2, dcp_rank=0, backend="mlx", stream=mx.cpu)
    options.update(config)
    with pytest.raises(LocalizationError):
        localize_mlx(mx.array([0]), mx.array([[0]]), mx.array([[0]]), **options)


@pytest.mark.parametrize("index,replacement", [
    (0, [0]), (0, mx.array([0.0])), (0, mx.array([[0]])),
    (0, mx.array([], dtype=mx.int32)), (0, mx.array([0, 0])),
    (1, mx.array([0])), (1, mx.zeros((1, 0), dtype=mx.int32)),
    (1, mx.array([[0]], dtype=mx.int64)),
    (2, mx.array([0])), (2, mx.array([[True]])),
    (2, mx.zeros((0, 1), dtype=mx.int32)),
    (2, mx.zeros((1, 4097), dtype=mx.int32)),
])
def test_array_validation(index, replacement):
    args = [mx.array([0]), mx.array([[0]]), mx.array([[0]])]
    args[index] = replacement
    with pytest.raises(LocalizationError):
        localize_mlx(*args, block_size=4, dcp_size=2, dcp_rank=0, backend="mlx", stream=mx.cpu)


def test_auto_respects_cpu_stream_and_restores_default():
    before = mx.default_stream(mx.default_device())
    out, counts = localize_mlx(mx.array([0]), mx.array([[2]]), mx.array([[0, 1]]),
                              block_size=1, dcp_size=2, dcp_rank=0, stream=mx.cpu)
    assert out.tolist() == [[2, -1]]
    assert counts.tolist() == [1]
    assert mx.default_stream(mx.default_device()) == before


def test_metal_rejects_cpu_stream():
    with pytest.raises(LocalizationError, match="Apple GPU"):
        localize_mlx(mx.array([0]), mx.array([[0]]), mx.array([[0]]), block_size=1,
                     dcp_size=1, dcp_rank=0, backend="metal", stream=mx.cpu)


def test_fixed_width_integral_geometry_does_not_overflow_validation():
    class WrappingInt(int):
        def __mul__(self, other):
            return 0  # Model fixed-width integer multiplication overflowing.

    with pytest.raises(LocalizationError, match="fit int32"):
        localize_mlx(mx.array([0]), mx.array([[0]]), mx.array([[0]]),
                     block_size=65536, dcp_size=WrappingInt(65536), dcp_rank=0,
                     dcp_interleave=WrappingInt(65536), backend="mlx", stream=mx.cpu)


def test_integral_scalars_are_normalized_through_kernel_specialization(execution):
    class ScalarBoolean:
        def __init__(self, value):
            self.value = value

        def __bool__(self):
            return self.value

    class IntegralScalar(int):
        # NumPy integers return np.bool_ here. Model that without adding a
        # NumPy dependency: Metal accepts builtin bool/int template values.
        def __gt__(self, other):
            return ScalarBoolean(super().__gt__(other))

    _check([0], [[2]], [[2, 1, 0, -1]], execution,
           block_size=IntegralScalar(2), dcp_size=IntegralScalar(2),
           dcp_rank=IntegralScalar(0), dcp_interleave=IntegralScalar(1))


@pytest.mark.metal
def test_compiled_metal_and_changing_inputs():
    if not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    req, table = mx.array([0]), mx.array([[2, 1]])
    function = mx.compile(lambda tokens: localize_mlx(
        req, table, tokens, block_size=2, dcp_size=2, dcp_rank=0, backend="metal"))
    for row in ([0, 1, 2, 3], [7, 6, 4, -1], [-1, -1, -1, -1]):
        out, counts = function(mx.array([row], dtype=mx.int32))
        assert (out.tolist(), counts.tolist()) == localize_reference(
            [0], [[2, 1]], [row], block_size=2, dcp_size=2, dcp_rank=0)


@pytest.mark.metal
def test_large_batch_constructs_without_overflowing_dispatch_dimensions():
    if not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    # Broadcast views keep this host-side launch-metadata regression small.
    # Do not evaluate: it checks the previous grid=(batch*256,1,1) TypeError
    # without allocating or executing millions of rows.
    batch = 2**23
    with mx.stream(mx.cpu):
        req = mx.broadcast_to(mx.array([0], dtype=mx.int32), (batch,))
        table = mx.array([[0]], dtype=mx.int32)
        tokens = mx.broadcast_to(mx.array([[0]], dtype=mx.int32), (batch, 1))
    out, counts = localize_mlx(
        req, table, tokens, block_size=1, dcp_size=2, dcp_rank=0, backend="metal"
    )
    assert out.shape == (batch, 1)
    assert counts.shape == (batch,)


def test_thread_local_stream_resolves_in_each_worker(execution):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    backend, device = execution
    stream = mx.new_thread_local_stream(device)
    ready = Barrier(3)

    def worker(index):
        before_device = mx.default_device()
        before_stream = mx.default_stream(before_device)
        with mx.stream(stream):
            own_stream = mx.default_stream(device)
            req = mx.array([0], dtype=mx.int32)
            table = mx.array([[index + 1]], dtype=mx.int32)
            tokens = mx.array([[2, 1, 0, -1]], dtype=mx.int32)
            mx.eval(req, table, tokens)
        ready.wait(timeout=10)
        out, counts = localize_mlx(
            req, table, tokens, block_size=2, dcp_size=2, dcp_rank=0,
            backend=backend, stream=stream,
        )
        mx.eval(out, counts)
        assert out.tolist() == [[(index + 1) * 2 + 1, (index + 1) * 2, -1, -1]]
        assert counts.tolist() == [2]
        assert mx.default_device() == before_device
        assert mx.default_stream(before_device) == before_stream
        return repr(own_stream)

    with ThreadPoolExecutor(max_workers=3) as pool:
        streams = list(pool.map(worker, range(3)))
    assert len(set(streams)) == 3
