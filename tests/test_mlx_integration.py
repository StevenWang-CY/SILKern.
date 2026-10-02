"""Sparse attention across logical KV shards on one device, without distribution.

This exercises localization's consumer contract: paged K/V gathers, duplicate
selections, prefix masks, and softmax recombination. It is a correctness test,
not a multi-device integration or a decode performance measurement.
"""

from __future__ import annotations

import math
from functools import partial

import pytest

pytestmark = pytest.mark.mlx
mx = pytest.importorskip("mlx.core")

from examples.mlx_sparse_attention import selected_attention  # noqa: E402


@pytest.fixture(params=[
    pytest.param(("mlx", mx.cpu), id="native-cpu"),
    pytest.param(("metal", mx.gpu), id="metal-gpu", marks=pytest.mark.metal),
])
def execution(request):
    backend, device = request.param
    if backend == "metal" and not mx.metal.is_available():
        pytest.skip("Apple Metal device unavailable")
    return backend, device


def _logical_kv(request, token):
    key = [math.sin((request + 1) * (token + 1) / (dimension + 2)) + 0.03 * token
           for dimension in range(4)]
    value = [math.cos((token + 1) / (dimension + 1)) + 0.4 * request + 0.01 * token
             for dimension in range(3)]
    return key, value


def _unsharded_attention(requests, rows, queries, global_limit):
    """Independent scalar attention over selected logical tokens, in float64."""
    expected = []
    for request, row, query in zip(requests, rows, queries, strict=True):
        selected = [_logical_kv(request, token) for token in row if 0 <= token < global_limit]
        if not selected:
            expected.append([0.0] * 3)
            continue
        logits = [sum(q * k for q, k in zip(query, key, strict=True)) / math.sqrt(len(query))
                  for key, _ in selected]
        maximum = max(logits)
        weights = [math.exp(logit - maximum) for logit in logits]
        denominator = sum(weights)
        expected.append([
            sum(weight * value[dimension] for weight, (_, value) in zip(weights, selected, strict=True))
            / denominator
            for dimension in range(3)
        ])
    return expected


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("dcp_size", [1, 2, 3])
@pytest.mark.parametrize("interleave", [1, 2])
def test_localized_kv_attention_matches_unsharded_selection(execution, dcp_size, interleave, compiled):
    backend, device = execution
    block_size, table_width, request_count = 4, 3, 3
    local_limit = block_size * table_width
    global_limit = local_limit * dcp_size
    requests = [2, 0, 1, 2, 0]
    rows = [
        [global_limit - 1, 0, 5, 3, 5, -1, global_limit, 1, 9, 2, -9],
        [7, 2, -1, 0, 4, 1, 8, 7, global_limit + 2, 6, 11],
        [-1, -7, global_limit, global_limit + 100, -2, -1, global_limit + 3,
         -8, global_limit, -1, -9],
        [0, 0, 0, -1, 0, -1, 0, -1, -3, 0, -1],
        [10, global_limit - 2, 3, 10, 2, -1, global_limit + 1, 5, 6, global_limit - 1, 0],
    ]
    queries = [[0.17 * (row + 1) - 0.11 * dimension for dimension in range(4)]
               for row in range(len(rows))]
    expected = _unsharded_attention(requests, rows, queries, global_limit)

    # The stream context covers every gather and attention operation as well as
    # localization. Both logical-shard layouts execute on this one device.
    with mx.stream(device):
        req = mx.array(requests, dtype=mx.int32)
        selections = mx.array(rows, dtype=mx.int32)
        query = mx.array(queries, dtype=mx.float32)
        shards = []
        for rank in range(dcp_size):
            # Nine logical request pages map to unique, fragmented physical
            # pages. Each rank has a different map; page zero stays poisoned.
            table = [[1 + (5 * (request * table_width + page) + 2 * rank) % 11
                      for page in range(table_width)] for request in range(request_count)]
            keys = [[float("nan")] * 4 for _ in range(12 * block_size)]
            values = [[float("nan")] * 3 for _ in range(12 * block_size)]
            for request in range(request_count):
                for local in range(local_limit):
                    group, lane = divmod(local, interleave)
                    global_token = (group * dcp_size + rank) * interleave + lane
                    physical = table[request][local // block_size] * block_size + local % block_size
                    keys[physical], values[physical] = _logical_kv(request, global_token)

            shards.append((mx.array(table, dtype=mx.int32),
                           mx.array(keys, dtype=mx.float32), mx.array(values, dtype=mx.float32)))

        function = partial(selected_attention, block_size=block_size,
                           dcp_interleave=interleave, backend=backend, stream=device)
        if compiled:
            function = mx.compile(function)
        result, rank_counts = function(query, req, selections, shards)
        mx.eval(result, rank_counts)
        for rank, counts in enumerate(rank_counts):
            expected_counts = [sum(0 <= token < global_limit
                                   and (token // interleave) % dcp_size == rank
                                   for token in row) for row in rows]
            assert counts.tolist() == expected_counts
            assert counts.tolist()[2] == 0
            if rank != 0:
                assert counts.tolist()[3] == 0  # Empty shard of a nonempty selection.

        # Reuse the actual compiled consumer with changing queries, requests,
        # and selections. Capturing fixture arrays would fail this check.
        queries2 = [[-q for q in row] for row in queries]
        requests2, rows2 = requests[::-1], rows[::-1]
        changed, _ = function(mx.array(queries2, dtype=mx.float32),
                              mx.array(requests2, dtype=mx.int32),
                              mx.array(rows2, dtype=mx.int32), shards)
        mx.eval(changed)
        expected2 = _unsharded_attention(requests2, rows2, queries2, global_limit)

    observed = result.tolist()
    assert all(math.isfinite(value) for row in observed for value in row)
    assert observed[2] == [0.0, 0.0, 0.0]
    for actual, wanted in zip(observed, expected, strict=True):
        assert actual == pytest.approx(wanted, rel=1e-5, abs=1e-6)
    for actual, wanted in zip(changed.tolist(), expected2, strict=True):
        assert actual == pytest.approx(wanted, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("shard_count", [1, 2])
def test_attention_masks_invalid_requests_and_poisoned_cache(execution, shard_count):
    backend, device = execution
    with mx.stream(device):
        shard = (mx.array([[0]], dtype=mx.int32),
                 mx.full((4, 2), float("nan")), mx.full((4, 3), float("nan")))
        out, counts = selected_attention(
            mx.ones((2, 2)), mx.array([-1, 1], dtype=mx.int32),
            mx.zeros((2, 3), dtype=mx.int32), [shard] * shard_count,
            block_size=4, backend=backend, stream=device,
        )
        assert out.tolist() == [[0.0] * 3] * 2
        assert all(count.tolist() == [0, 0] for count in counts)
