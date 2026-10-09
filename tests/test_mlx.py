"""Oracle-backed MLX regressions on CPU and Apple Metal; no CUDA execution."""

from __future__ import annotations

import random
import threading

import pytest

import silkern.mlx as silkern_mlx
from silkern import MAX_ROW_WIDTH, LocalizationError, localize_mlx, localize_reference

pytestmark = pytest.mark.mlx
mx = pytest.importorskip("mlx.core")

# (backend, device) pairs that localize_mlx accepts, including auto selection.
EXECUTION_PATHS = [
    pytest.param("auto", "cpu", id="auto-cpu"),
    pytest.param("mlx", "cpu", id="mlx-cpu"),
    pytest.param("auto", "gpu", id="auto-gpu", marks=pytest.mark.metal),
    pytest.param("mlx", "gpu", id="mlx-gpu", marks=pytest.mark.metal),
    pytest.param("metal", "gpu", id="metal-gpu", marks=pytest.mark.metal),
]


def _device(name):
    if name == "gpu" and not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    return mx.cpu if name == "cpu" else mx.gpu


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


# Divisors for the multiply-high division: every small one, every page size
# that is not a power of two up to 2**28 in steps of seven, and the extremes.
DIVISORS = sorted(
    set(range(2, 258))
    | {7 * 2**k for k in range(28)}
    | {2**k + 1 for k in range(2, 31)}
    | {2**k - 1 for k in range(2, 32)}
    | {1_000_003, 715_827_882, 2**28, 2**30 + 7}
)


def _hard_numerators(d):
    """Numerators in [0, 2**31) where a too-short multiply-high shift goes wrong first."""
    top = 2**31 - 1
    worst = top - ((top - (d - 1)) % d)  # the largest n < 2**31 with n % d == d - 1
    candidates = {0, 1, d - 1, d, d + 1, worst, worst - d, top, top - 1, (top // d) * d,
                  2**30 - 1, 2**30, 2**30 + d - 1}
    return sorted(n for n in candidates if 0 <= n <= top)


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.metal)])
def test_division_by_multiplication_is_exact_to_the_int32_limit(device):
    stream = mx.default_stream(_device(device))
    for d in DIVISORS:
        numerators = _hard_numerators(d)
        quotient, remainder = silkern_mlx._divmod_nonnegative(
            mx, stream, mx.array(numerators, dtype=mx.int32), d
        )
        mx.eval(quotient, remainder)
        expected = ([n // d for n in numerators], [n % d for n in numerators])
        assert (quotient.tolist(), remainder.tolist()) == expected, d


def test_division_bound_reaches_the_output(execution):
    """The largest token on the last of seven ranks, with 2**28-slot pages."""
    _check([0], [[0, 3]], [[2**31 - 3, 5]], execution,
           block_size=2**28, dcp_size=7, dcp_rank=6)


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


@pytest.mark.parametrize("requests,table_width", [
    (1, 1), (1, 300), (2, 2), (3, 17), (5, 18), (9, 64), (40, 5),
])
def test_block_table_shapes_read_at_run_time_stay_exact(execution, requests, table_width):
    # Metal reads the batch, row width and table shape from the arrays, so
    # every shape below shares a few compiled kernels instead of one each.
    rng = random.Random(requests * 1009 + table_width)
    geometry = dict(block_size=4, dcp_size=2, dcp_rank=1, dcp_interleave=2)
    table = [[rng.randrange(-3, 1 << 20) for _ in range(table_width)] for _ in range(requests)]
    limit = geometry["block_size"] * table_width * 2 + 9
    for width in (1, 257, 1000, 4096):
        req = [requests - 1, 0, rng.randrange(requests)]
        rows = [[rng.randrange(-3, limit) for _ in range(width)] for _ in req]
        _check(req, table, rows, execution, **geometry)


@pytest.mark.metal
def test_one_metal_specialization_serves_every_batch_and_table_shape(monkeypatch):
    """Only the per-thread chunk length and scalar geometry select a kernel.

    MLX compiles one Metal library per distinct template. Table dimensions
    and the row width used to be template constants, so every new request
    count or page count in a serving loop paid a fresh compilation.
    """
    if not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    kernel = silkern_mlx._metal_kernel()
    templates = set()

    def recording_kernel(**kwargs):
        templates.add(tuple(kwargs["template"]))
        return kernel(**kwargs)

    monkeypatch.setattr(silkern_mlx, "_metal_kernel", lambda: recording_kernel)
    rng = random.Random(4)
    geometry = dict(block_size=8, dcp_size=2, dcp_rank=1)
    # Widths 257..512 all give each thread two elements: one specialization.
    for requests, table_width, batch, width in [
        (1, 1, 1, 257), (2, 3, 9, 300), (5, 18, 3, 511), (8, 33, 16, 512),
        (64, 7, 2, 400), (3, 255, 8, 260),
    ]:
        table = [[rng.randrange(-2, 1 << 16) for _ in range(table_width)]
                 for _ in range(requests)]
        req = [rng.randrange(requests) for _ in range(batch)]
        limit = geometry["block_size"] * table_width * 2 + 16
        rows = [[rng.randrange(-4, limit) for _ in range(width)] for _ in range(batch)]
        out, counts = localize_mlx(
            *(mx.array(value, dtype=mx.int32) for value in (req, table, rows)),
            **geometry, backend="metal", stream=mx.gpu,
        )
        assert (out.tolist(), counts.tolist()) == localize_reference(req, table, rows, **geometry)
    assert len(templates) == 1, templates


# Each step is (requests, table_width, request ids, rows, width). Request r owns
# pages 100 * (r + 1) + p, so a stale table shape reads another request's page.
SHAPE_SEQUENCES = {
    "table_grows": [(2, 6, 2, 2, 3), (2, 9, 2, 2, 3), (2, 12, 2, 2, 3)],
    "width_grows": [(2, 6, 2, 2, 3), (2, 6, 2, 2, 4), (2, 6, 2, 2, 5)],
    "width_then_table": [(2, 6, 2, 2, 3), (2, 6, 2, 2, 4), (2, 9, 2, 2, 4)],
    "batch_grows": [(2, 6, 2, 2, 3), (2, 6, 3, 3, 3), (2, 6, 5, 5, 3)],
    "requests_grow": [(2, 6, 2, 2, 3), (3, 6, 2, 2, 3), (5, 6, 2, 2, 3)],
    "width_beyond_limit": [(2, 6, 2, 2, 3), (2, 6, 2, 2, 4), (2, 6, 2, 2, MAX_ROW_WIDTH + 1)],
    "batch_mismatch": [(2, 6, 2, 2, 3), (2, 6, 2, 2, 4), (2, 6, 1, 3, 4)],
}
COMPILE_GEOMETRY = dict(block_size=4, dcp_size=2, dcp_rank=0)


def _shape_step(requests, table_width, request_rows, rows, width):
    table = [[100 * (request + 1) + page for page in range(table_width)]
             for request in range(requests)]
    # The first row always names the newest request.
    req = [(requests - 1 - row) % requests for row in range(request_rows)]
    limit = 2 * COMPILE_GEOMETRY["block_size"] * table_width + 3
    tokens = [[-1 if col == 1 else (5 * col + 7 * row) % limit for col in range(width)]
              for row in range(rows)]
    return req, table, tokens


def _eager_accepts(requests, table_width, request_rows, rows, width):
    return request_rows == rows and width <= MAX_ROW_WIDTH


def _compiled(backend, stream, **options):
    return mx.compile(lambda r, t, x: localize_mlx(
        r, t, x, **COMPILE_GEOMETRY, backend=backend, stream=stream), **options)


@pytest.mark.parametrize("sequence", sorted(SHAPE_SEQUENCES))
@pytest.mark.parametrize("backend,device", EXECUTION_PATHS)
def test_shapeless_compilation_is_exact_or_refused(backend, device, sequence):
    """A shapeless trace is replayed for new shapes without running Python.

    Each call must equal the oracle or raise; it must never read stale
    geometry or accept a shape that eager validation rejects.
    """
    compiled = _compiled(backend, _device(device), shapeless=True)
    for step in SHAPE_SEQUENCES[sequence]:
        req, table, tokens = _shape_step(*step)
        try:
            out, counts = compiled(*(mx.array(value, dtype=mx.int32)
                                     for value in (req, table, tokens)))
            mx.eval(out, counts)
        except ValueError:  # Includes LocalizationError; refusing is safe.
            continue
        assert _eager_accepts(*step), f"accepted geometry eager validation rejects: {step}"
        assert (out.tolist(), counts.tolist()) == localize_reference(
            req, table, tokens, **COMPILE_GEOMETRY), step


@pytest.mark.parametrize("backend,device", EXECUTION_PATHS)
def test_shapeless_compilation_is_refused_on_its_first_call(backend, device):
    # Fail fast, before a deployment ever sees a second shape.
    compiled = _compiled(backend, _device(device), shapeless=True)
    req, table, tokens = _shape_step(2, 6, 2, 2, 3)
    with pytest.raises(ValueError, match="cannot infer output shapes"):
        compiled(*(mx.array(value, dtype=mx.int32) for value in (req, table, tokens)))


@pytest.mark.parametrize("sequence", sorted(SHAPE_SEQUENCES))
@pytest.mark.parametrize("backend,device", EXECUTION_PATHS)
def test_ordinary_compilation_retraces_and_revalidates_new_shapes(backend, device, sequence):
    compiled = _compiled(backend, _device(device))
    for step in SHAPE_SEQUENCES[sequence]:
        req, table, tokens = _shape_step(*step)
        arrays = [mx.array(value, dtype=mx.int32) for value in (req, table, tokens)]
        if not _eager_accepts(*step):
            with pytest.raises(LocalizationError):
                compiled(*arrays)
            continue
        out, counts = compiled(*arrays)
        assert (out.tolist(), counts.tolist()) == localize_reference(
            req, table, tokens, **COMPILE_GEOMETRY), step


STREAM_GEOMETRY = dict(block_size=64, dcp_size=2, dcp_rank=0)
STREAM_CASE = ([1, 0, 1, 0], [[11, 2, 7, 5], [3, 9, 1, 8]],
               [[8, 5, 130, 7, -1, 262, 64, 0, 3], [2, 128, 3, 258, 128, -1, 9, 0, 511]] * 2)
STREAM_KINDS = ("none", "device_type", "device", "default_stream", "new_stream", "thread_local")


def _stream_argument(kind, device):
    return {
        "none": lambda: None,
        "device_type": lambda: device,
        "device": lambda: mx.Device(device),
        "default_stream": lambda: mx.default_stream(device),
        "new_stream": lambda: mx.new_stream(device),
        "thread_local": lambda: mx.new_thread_local_stream(device),
    }[kind]()


def _default_streams():
    devices = (mx.cpu, mx.gpu) if mx.metal.is_available() else (mx.cpu,)
    return [mx.default_stream(device) for device in devices]


@pytest.mark.parametrize("kind", STREAM_KINDS)
@pytest.mark.parametrize("backend,device", EXECUTION_PATHS)
def test_every_operation_runs_on_the_selected_stream(backend, device, kind):
    """No operation may fall back to MLX's process-wide default device.

    MLX accepts a default device that has no streams; any operation created
    without the selected stream then fails at once, before anything runs.
    ``stream=None`` with ``auto`` or ``mlx`` documents that it uses the default
    device, so that case sets it to the requested one. The default device and
    this thread's default streams must be unchanged afterwards.
    """
    target = _device(device)
    argument = _stream_argument(kind, target)
    selects_default = kind == "none" and backend != "metal"
    default_device = target if selects_default else mx.Device(mx.cpu, 99)
    streams = _default_streams()
    initial = mx.default_device()
    try:
        mx.set_default_device(default_device)
        out, counts = localize_mlx(*(mx.array(value, dtype=mx.int32) for value in STREAM_CASE),
                                   **STREAM_GEOMETRY, backend=backend, stream=argument)
        mx.eval(out, counts)
        assert mx.default_device() == default_device
    finally:
        mx.set_default_device(initial)
    assert (out.tolist(), counts.tolist()) == localize_reference(*STREAM_CASE, **STREAM_GEOMETRY)
    assert _default_streams() == streams


@pytest.mark.metal
@pytest.mark.parametrize("kind", ["device_type", "thread_local"])
@pytest.mark.parametrize("first,second", [
    (("mlx", "cpu"), ("mlx", "gpu")),
    (("mlx", "cpu"), ("metal", "gpu")),
    (("mlx", "gpu"), ("mlx", "cpu")),
], ids=["cpu-with-gpu", "cpu-with-metal", "gpu-with-cpu"])
def test_concurrent_calls_on_other_devices_keep_their_own_streams(monkeypatch, first, second,
                                                                  kind):
    """Force the interleaving that broke the former ``with mx.stream`` block.

    A enters localize_mlx and pauses, B enters and pauses, A builds its graph
    and returns, then B builds. Each thread reads the process default device
    while the other one is inside its call: neither call may change it, even
    temporarily. Afterwards it, and each thread's default streams, must be
    unchanged, and both results exact.
    """
    if not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    a_inside, b_inside, a_returned = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def pausing(original):
        def paused(*args, **kwargs):
            name = threading.current_thread().name
            if name in ("A", "B"):
                (a_inside if name == "A" else b_inside).set()
                waited = (b_inside if name == "A" else a_returned).wait(10)
                seen[name] = (waited, mx.default_device())
            return original(*args, **kwargs)
        return paused

    monkeypatch.setattr(silkern_mlx, "_localize_native", pausing(silkern_mlx._localize_native))
    monkeypatch.setattr(silkern_mlx, "_metal_kernel", pausing(silkern_mlx._metal_kernel))
    streams = {device: _stream_argument(kind, mx.cpu if device == "cpu" else mx.gpu)
               for device in ("cpu", "gpu")}
    expected = localize_reference(*STREAM_CASE, **STREAM_GEOMETRY)
    results, errors = {}, {}

    def body(name, backend, device):
        try:
            before = _default_streams()
            if name == "B":
                a_inside.wait(10)
            out, counts = localize_mlx(*(mx.array(value, dtype=mx.int32) for value in STREAM_CASE),
                                       **STREAM_GEOMETRY, backend=backend, stream=streams[device])
            if name == "A":
                a_returned.set()
            mx.eval(out, counts)
            results[name] = (out.tolist(), counts.tolist())
            assert _default_streams() == before
        except BaseException as exc:  # Reported by the test thread.
            errors[name] = exc
        finally:
            for event in (a_inside, b_inside, a_returned):
                event.set()  # Never leave the other thread waiting after a failure.

    initial = mx.default_device()
    threads = [threading.Thread(target=body, args=("A", *first), name="A"),
               threading.Thread(target=body, args=("B", *second), name="B")]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        final = mx.default_device()
    finally:
        mx.set_default_device(initial)
    assert not errors, errors
    assert seen == {"A": (True, initial), "B": (True, initial)}, seen
    assert final == initial
    assert results == {"A": expected, "B": expected}
