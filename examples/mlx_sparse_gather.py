"""A localized sparse KV gather across two logical shards on one Apple device.

Run from the checkout: ``python -m examples.mlx_sparse_gather``.
This is an integration example, not a distributed runtime or decode benchmark.
"""

from __future__ import annotations

import mlx.core as mx

from silkern import localize_mlx


def main() -> None:
    # Each logical shard uses a different fragmented physical page map.
    tables = ([[2, 0]], [[1, 3]])
    rows = [[7, 0, 14, 3, -1, 6, 1, 7]]  # selector order, duplicates, a sentinel
    selections = mx.array(rows, dtype=mx.int32)
    request = mx.array([0], dtype=mx.int32)
    gathered_sum = mx.zeros((2,), dtype=mx.float32)
    for rank, table in enumerate(tables):
        # Build a toy KV cache; real integrations supply their existing cache.
        slots = [[0.0, 0.0] for _ in range(16)]
        for local in range(8):
            global_token = 2 * local + rank
            physical = table[0][local // 4] * 4 + local % 4
            slots[physical] = [float(global_token), global_token + 0.5]
        cache = mx.array(slots, dtype=mx.float32)
        out, counts = localize_mlx(request, mx.array(table, dtype=mx.int32), selections,
                                   block_size=4, dcp_size=2, dcp_rank=rank)
        # Compacted DCP>1 output: only this prefix belongs to the consumer.
        valid = mx.arange(selections.shape[1])[None, :] < counts[:, None]
        safe_slots = mx.where(valid, out, 0)
        gathered = mx.take(cache, safe_slots, axis=0)
        gathered_sum = gathered_sum + mx.sum(mx.where(valid[..., None], gathered, 0), axis=(0, 1))
        print(f"rank {rank}: slots={out.tolist()}, counts={counts.tolist()}")

    selected = [token for token in rows[0] if token >= 0]
    expected = mx.array([sum(selected), sum(token + 0.5 for token in selected)])
    assert mx.array_equal(gathered_sum, expected).item()
    print(f"Recombined sparse gather: {gathered_sum.tolist()} (matches logical selection)")


if __name__ == "__main__":
    main()
