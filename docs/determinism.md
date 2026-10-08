# Why stable localization matters

[README](../README.md) · [Contract](contract.md) · [Architecture](architecture.md) · [Evidence](evidence.md)

An atomic-reservation converter can preserve the selected multiset and count
while changing the order of the returned prefix. SILKern strengthens that
boundary by preserving the selector's relative order.

<picture>
  <source media="(max-width: 767px)" srcset="../assets/fig-problem-narrow.svg">
  <img src="../assets/fig-problem.svg" width="100%" alt="Three replays of the same four tile groups: atomic reservation emits a different group order on each replay, while offsets from an exclusive scan keep the input order every time.">
</picture>

**Tile order under repeated execution.** The replay arrays are illustrative: atomic reservation claims output offsets in completion order and can permute whole tile groups, while offsets from an exclusive scan preserve input order. Both return the same eight values and count. Each tile holds two ordered slots, and curves trace the first replay. The separate [historical two-B200 record](../evidence/06-order-instability-b200/orders.json) measured 17–20 atomic orders per 20 replays and one stable order.

## Where the order changes

A tiled converter can compute a local prefix within each tile and reserve space
with an atomic counter:

```text
base = atomic_add(counter[row], tile_count)
out[row][base + local_position] = physical
```

The atomic makes reservations disjoint; it does not define which tile reserves
first. Scheduling determines tile bases, so the final prefix can vary even
when every input byte is unchanged. At width 2048 and tile size 128, a row has
16 tiles whose reservations can arrive in different orders.

The historical two-B200 artifact records **17–20 distinct consumed orders** per
20 replays of identical inputs, while the stable converter returns one order.
This is a recorded result about that experiment, not a measurement from the
current development machine. See [raw order records](../evidence/06-order-instability-b200/orders.json).

## Why the consumer can care

Floating-point addition is not associative. Changing the order of KV rows can
change an attention reduction's last bits, and a near-tie between logits can
then change a chosen token. Whether that happens depends on the consumer,
numerics, and input; it is not inevitable on every decode step.

Stable localization is useful when comparing sparse-attention implementations,
replaying a failure, debugging cached versus uncached paths, or auditing rollout
and evaluation reproducibility. It removes one variable from those comparisons.
The repository does not quantify RL training bias, evaluation-score drift, or
prefix-cache effects attributable to this converter.

## Deterministic destinations

SILKern computes a survivor's location from input order:

```text
destination[j] = sum(valid[0:j+1]) - 1
```

The rowwise CUDA implementation computes a whole-row scan. The hierarchical
implementation adds deterministic tile offsets to tile-local positions. Apple
MLX offers a custom Metal scan and a compositional array implementation. Their
valid-input output contract is the same.

Alternatives have different tradeoffs:

| Approach | Consequence |
|---|---|
| Sort physical indices | Defines a new order; does not restore selector order |
| Atomic reservation per element | Still scheduling-dependent, with more reservations |
| Serialize reservations | Can preserve order at the expense of parallel work |
| Deterministic prefix positions | Preserves selector order without racing for output segments |

Sorting or staging can be designed with preallocated workspace; allocation is
not an inherent impossibility. The reason to use a stable prefix is that it
expresses the desired order directly. PyTorch's deterministic-algorithms setting
does not automatically rewrite a custom Triton converter.

## What this guarantee covers

For identical valid integer inputs and geometry, localization returns identical
values, counts, and order. This does not promise bitwise-identical model output:
attention and GEMM reductions, collective order, batch composition, and other
components may still vary. Preserving selector order is a reproducibility
property, not evidence of improved model quality.

The performance cost is empirical. Historical rowwise converter measurements
are favorable, while hierarchical has more launch overhead and one rowwise
complete-step result regresses. See [dispatch](dispatch.md) and
[evidence](evidence.md) rather than assuming stable ordering is always faster.
