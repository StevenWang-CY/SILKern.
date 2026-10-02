"""Watch the atomic converter change its mind, then watch silkern not.

    python -m bench.order_instability

Replays each converter N times on identical input, checks results against the
oracle, and counts distinct output orders. The atomic baseline may change
survivor order while preserving the valid multiset and count; this can affect
downstream floating-point reductions.

The atomic baseline may return multiple orders, depending on scheduling and
geometry; a run observing one order does not prove it is stable. Both silkern
implementations must return exactly one order.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from bench.bench_converter import _case, _check_results, _validate_options
from silkern import (
    DEFAULT_TILE_SIZE,
    LocalizationError,
    localize_hierarchical,
    localize_reference,
    localize_rowwise,
    workspace_shapes,
)


def _digest(tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()[:16]


def _set_digest(tensor, counts) -> str:
    """Order-insensitive digest of the valid prefix, per row."""
    hasher = hashlib.sha256()
    rows = tensor.cpu().tolist()
    for row, count in zip(rows, counts.cpu().tolist(), strict=True):
        hasher.update(repr(sorted(row[:count])).encode())
    return hasher.hexdigest()[:16]


def _summarize_observations(observations) -> tuple[int, str]:
    all_sets = set().union(*(result[1] for result in observations.values()))
    all_counts = set().union(*(result[2] for result in observations.values()))
    stable_orders = [len(result[0]) for name, result in observations.items() if name != "atomic_baseline"]
    if len(all_sets) != 1 or len(all_counts) != 1 or any(count != 1 for count in stable_orders):
        return 1, "FAIL: converters disagreed on sets/counts or a stable converter changed order."
    if len(observations["atomic_baseline"][0]) > 1:
        return 0, "Same set and count across all converters; atomic output order varied in this run."
    return 0, (
        "Same set and count; no atomic order variation observed in this run.\n"
        "One observed order does not establish deterministic atomic reservation."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--dcp-rank", type=int, default=0)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        _validate_options(args, with_atomic=True)
    except LocalizationError as exc:
        parser.error(str(exc))

    try:
        import torch
    except ModuleNotFoundError:
        print("torch is not installed; install silkern[gpu]", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 2

    from bench.atomic_baseline import launch_atomic_baseline

    req_ids, block_table, rows = _case(
        args.width, args.batch, args.block_size, args.dcp_size, args.seed
    )

    dev = "cuda"
    req = torch.tensor(req_ids, dtype=torch.int32, device=dev)
    table = torch.tensor(block_table, dtype=torch.int32, device=dev)
    tokens = torch.tensor(rows, dtype=torch.int32, device=dev)
    out = torch.empty_like(tokens)
    counts = torch.empty(args.batch, dtype=torch.int32, device=dev)
    scratch = torch.empty(args.batch, dtype=torch.int32, device=dev)
    ws = {
        name: torch.empty(shape, dtype=torch.int32, device=dev)
        for name, shape in workspace_shapes(args.batch, args.width).items()
    }
    common = dict(
        block_size=args.block_size, dcp_size=args.dcp_size, dcp_rank=args.dcp_rank
    )
    expected_out, expected_counts = localize_reference(req_ids, block_table, rows, **common)

    arms = {
        "atomic_baseline": lambda: launch_atomic_baseline(
            req, table, tokens, out, counts, scratch, **common
        ),
        "silkern row_stable": lambda: localize_rowwise(
            req, table, tokens, out, counts, **common
        ),
        "silkern hierarchical": lambda: localize_hierarchical(
            req,
            table,
            tokens,
            out,
            counts,
            ws["mapped"],
            ws["local_positions"],
            ws["tile_counts"],
            ws["tile_offsets"],
            tile_size=DEFAULT_TILE_SIZE,
            **common,
        ),
    }

    print(f"device   : {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"geometry : width={args.width} batch={args.batch} "
        f"dcp={args.dcp_rank}/{args.dcp_size}, {args.replays} replays of identical input\n"
    )
    print(f"  {'converter':22s} {'distinct orders':>16s} {'distinct sets':>15s} {'counts':>8s}")
    print(f"  {'-' * 22} {'-' * 16:>16s} {'-' * 15:>15s} {'-' * 8:>8s}")

    observations = {}
    for name, launch in arms.items():
        order_digests: set[str] = set()
        set_digests: set[str] = set()
        count_digests: set[str] = set()
        for _ in range(args.replays):
            try:
                launch()
                torch.cuda.synchronize()
                _check_results(name, out.cpu().tolist(), counts.cpu().tolist(),
                               expected_out, expected_counts)
            except (LocalizationError, RuntimeError) as exc:
                print(f"order qualification failed: {exc}", file=sys.stderr)
                return 1
            order_digests.add(_digest(out))
            set_digests.add(_set_digest(out, counts))
            count_digests.add(_digest(counts))
        print(
            f"  {name:22s} {len(order_digests):>16d} {len(set_digests):>15d} "
            f"{len(count_digests):>8d}"
        )
        observations[name] = (order_digests, set_digests, count_digests)

    code, summary = _summarize_observations(observations)
    print(f"\n{summary}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
