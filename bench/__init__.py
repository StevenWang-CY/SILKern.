"""Benchmarks and demonstrations. Run from the repository root:

    python -m bench.order_instability             # CUDA: atomic vs stable order
    python -m bench.bench_converter --with-atomic # CUDA: converter latency
    python -m bench.bench_mlx --compiled          # Apple: Metal vs MLX localization
    python -m bench.bench_mlx_attention           # Apple: complete selected attention
    python -m bench.summarize_mlx FILES...        # Apple: recompute a session summary

These are diagnostics, not experiments: one host, one process, no statistical
framing. They exist so you can see the behavior on your own hardware rather than
take this repository's numbers on trust.
"""
