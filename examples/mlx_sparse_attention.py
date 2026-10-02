"""A complete selected-attention consumer on one Apple device.

Run ``python -m examples.mlx_sparse_attention --backend metal --device gpu``.
The float32 single-head example uses logical cache shards, not distributed
collectives. It is instructional code, not a SILKern public attention API.
"""

from __future__ import annotations

import argparse
import math

import mlx.core as mx

from silkern import localize_mlx


def selected_attention(
    query, req_ids, token_indices, shards, *, block_size, dcp_interleave=1,
    backend="auto", stream=None,
):
    """Return selected attention and per-shard counts as lazy MLX arrays.

    ``query`` is float32 (batch, key_dim); each shard is a tuple of its int32
    page table, float32 keys (slots, key_dim), and values (slots, value_dim).
    Queries and keys/values for every selected, populated page must be finite;
    dot products and weighted sums must remain representable in float32. The
    caller must guarantee all translated slots address the corresponding cache;
    negative page entries allowed by localization are not valid cache addresses.

    Invalid requests and empty selections return zeros. Duplicate selections
    retain their multiplicity. One logical shard uses the DCP-1 hole layout;
    multiple shards use compacted prefixes. Numerators and denominators share
    one maximum, so shard contributions are combined before normalization.
    """
    if not shards:
        raise ValueError("at least one logical cache shard is required")
    if query.ndim != 2 or query.dtype != mx.float32 or min(query.shape) < 1:
        raise ValueError("query must be a nonempty float32 (batch, key_dim) array")
    if token_indices.ndim != 2 or token_indices.shape[0] != query.shape[0]:
        raise ValueError("query and selections must have the same batch size")
    value_dim = None
    for _, keys, values in shards:
        if (keys.ndim != 2 or values.ndim != 2 or keys.dtype != mx.float32
                or values.dtype != mx.float32 or min(*keys.shape, *values.shape) < 1
                or keys.shape[0] != values.shape[0] or keys.shape[1] != query.shape[1]):
            raise ValueError("each shard needs matching nonempty float32 key/value caches")
        if value_dim is not None and values.shape[1] != value_dim:
            raise ValueError("all shards must have the same value dimension")
        value_dim = values.shape[1]

    device = stream if stream is not None else (
        mx.gpu if backend == "metal" else mx.default_device()
    )
    with mx.stream(device):
        columns = mx.arange(token_indices.shape[1], dtype=mx.int32)[None, :]
        shard_data, rank_counts = [], []
        for rank, (table, keys, values) in enumerate(shards):
            localized, counts = localize_mlx(
                req_ids, table, token_indices, block_size=block_size,
                dcp_size=len(shards), dcp_rank=rank, dcp_interleave=dcp_interleave,
                backend=backend, stream=device,
            )
            if len(shards) == 1:
                # DCP-1 deliberately bypasses compaction: counts alone cannot
                # identify holes. Derive validity from the selection domain.
                valid = ((token_indices >= 0)
                         & (mx.maximum(token_indices, 0) // int(block_size) < table.shape[1])
                         & (req_ids[:, None] >= 0) & (req_ids[:, None] < table.shape[0]))
            else:
                valid = columns < counts[:, None]
            safe_slots = mx.where(valid, localized, 0)
            gathered_keys = mx.take(keys, safe_slots, axis=0)
            gathered_values = mx.take(values, safe_slots, axis=0)
            # The placeholder slot can contain NaN. Mask values explicitly;
            # zero weights alone are insufficient because NaN * 0 is NaN.
            gathered_keys = mx.where(valid[..., None], gathered_keys, 0)
            gathered_values = mx.where(valid[..., None], gathered_values, 0)
            logits = mx.sum(query[:, None, :] * gathered_keys, axis=-1) / math.sqrt(query.shape[1])
            logits = mx.where(valid, logits, -float("inf"))
            shard_data.append((valid, logits, gathered_values))
            rank_counts.append(counts)

        common_max = mx.max(mx.stack([mx.max(logits, axis=1)
                                     for _, logits, _ in shard_data]), axis=0)
        # Avoid -inf - -inf for a row with no selected tokens on any shard.
        common_max = mx.where(mx.isfinite(common_max), common_max, 0)
        numerators, denominators = [], []
        for valid, logits, gathered_values in shard_data:
            weights = mx.where(valid, mx.exp(logits - common_max[:, None]), 0)
            numerators.append(mx.sum(weights[..., None] * gathered_values, axis=1))
            denominators.append(mx.sum(weights, axis=1))
        numerator = mx.sum(mx.stack(numerators), axis=0)
        denominator = mx.sum(mx.stack(denominators), axis=0)
        output = numerator / mx.where(denominator > 0, denominator, 1)[:, None]
        return output, tuple(rank_counts)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("auto", "mlx", "metal"), default="auto")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args(argv)
    if args.backend == "metal" and args.device == "cpu":
        parser.error("the Metal backend requires --device gpu")
    if args.device == "gpu" and not mx.metal.is_available():
        parser.error("Apple Metal is unavailable; try --device cpu --backend mlx")

    device = mx.cpu if args.device == "cpu" else mx.gpu
    with mx.stream(device):
        logical_keys = [[math.sin(token + 1), math.cos(token / 3)] for token in range(16)]
        logical_values = [[token / 16, math.sin(token / 5), 1.0] for token in range(16)]
        rows = [[7, 0, 14, 3, -1, 6, 1, 7], [-1] * 8, [0, 2, 4, -1, 0, 6, 2, -1]]
        queries = [[0.5, -0.3], [0.2, 0.1], [-0.4, 0.8]]
        shards = []
        for rank, table in enumerate(([[2, 1]], [[1, 3]])):
            keys = [[float("nan")] * 2 for _ in range(16)]
            values = [[float("nan")] * 3 for _ in range(16)]
            for local in range(8):
                slot = table[0][local // 4] * 4 + local % 4
                keys[slot], values[slot] = logical_keys[2 * local + rank], logical_values[2 * local + rank]
            shards.append((mx.array(table, dtype=mx.int32),
                           mx.array(keys, dtype=mx.float32), mx.array(values, dtype=mx.float32)))
        output, counts = selected_attention(
            mx.array(queries, dtype=mx.float32), mx.array([0, 0, 0], dtype=mx.int32),
            mx.array(rows, dtype=mx.int32), shards, block_size=4,
            backend=args.backend, stream=device,
        )
        mx.eval(output, counts)

    expected = []
    for query, row in zip(queries, rows, strict=True):
        selected = [token for token in row if 0 <= token < 16]
        if not selected:
            expected.append([0.0] * 3)
            continue
        logits = [sum(q * k for q, k in zip(query, logical_keys[t], strict=True))
                  / math.sqrt(2) for t in selected]
        weights = [math.exp(logit - max(logits)) for logit in logits]
        expected.append([sum(w * logical_values[t][d] for w, t in zip(weights, selected, strict=True))
                         / sum(weights) for d in range(3)])
    observed = output.tolist()
    if not all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
               for actual, wanted in zip(observed, expected, strict=True)
               for a, b in zip(actual, wanted, strict=True)):
        raise RuntimeError("selected attention differs from the unsharded reference")
    print(f"Logical shard counts: {[count.tolist() for count in counts]}")
    print(f"Selected attention: {observed}")
    print("Matches the unsharded reference, including duplicate tokens and an empty row.")


if __name__ == "__main__":
    main()
