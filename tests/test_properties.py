"""Seeded property tests of the localization contract, runnable anywhere.

The oracle is the normative definition, so these tests check it two ways that
do not share its code: an independent restatement of the five contract steps,
and metamorphic properties that any correct localization must satisfy whatever
its implementation. Every case is drawn from a fixed seed; a failure message
names the case so it can be replayed exactly.

``SILKERN_PROPERTY_CASES`` raises the number of random cases (default 400).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace

import pytest

from silkern import localize_reference

CASES = int(os.environ.get("SILKERN_PROPERTY_CASES", "400"))


@dataclass(frozen=True)
class Case:
    req_ids: list[int]
    block_table: list[list[int]]
    rows: list[list[int]]
    block_size: int
    dcp_size: int
    dcp_rank: int
    dcp_interleave: int
    compact_valid_to_front: bool

    def call(self, **changes) -> tuple[list[list[int]], list[int]]:
        case = replace(self, **changes)
        return localize_reference(
            case.req_ids,
            case.block_table,
            case.rows,
            block_size=case.block_size,
            dcp_size=case.dcp_size,
            dcp_rank=case.dcp_rank,
            dcp_interleave=case.dcp_interleave,
            compact_valid_to_front=case.compact_valid_to_front,
        )


def _random_case(rng: random.Random, *, nonnegative_pages: bool = False) -> Case:
    interleave = rng.choice((1, 1, 2, 3, 4, 8))
    block_size = interleave * rng.choice((1, 2, 3, 4, 5, 8, 16))
    dcp_size = rng.choice((1, 1, 2, 2, 3, 4, 5, 8))
    requests = rng.randint(1, 4)
    table_width = rng.randint(1, 6)
    low = 0 if nonnegative_pages else -3
    block_table = [[rng.randrange(low, 400) for _ in range(table_width)] for _ in range(requests)]
    # Tokens cover every branch: negative, owned and foreign, in and beyond the
    # table, duplicates, and integers far beyond int32 (the oracle is unbounded).
    covered = block_size * table_width * dcp_size
    width = rng.randint(1, 48)
    rows = []
    for _ in range(rng.randint(1, 4)):
        row: list[int] = []
        for _ in range(width):
            draw = rng.random()
            if draw < 0.12:
                row.append(rng.choice((-1, -2, -(2**40))))
            elif draw < 0.2 and row:
                row.append(rng.choice(row))
            elif draw < 0.28:
                row.append(rng.randrange(covered, 2 * covered + 1))
            elif draw < 0.3:
                row.append(2**40 + rng.randrange(1000))
            else:
                row.append(rng.randrange(covered))
        rows.append(row)
    return Case(
        req_ids=[rng.randrange(requests) for _ in rows],
        block_table=block_table,
        rows=rows,
        block_size=block_size,
        dcp_size=dcp_size,
        dcp_rank=rng.randrange(dcp_size),
        dcp_interleave=interleave,
        compact_valid_to_front=rng.random() < 0.7,
    )


def _restated(case: Case) -> tuple[list[list[int | None]], list[list[bool]]]:
    """The contract restated independently: (value, validity) per column."""
    values, validity = [], []
    for request, row in zip(case.req_ids, case.rows, strict=True):
        table = case.block_table[request]
        row_values: list[int | None] = []
        row_valid: list[bool] = []
        for token in row:
            value, valid = None, False
            if token >= 0:
                group, lane = divmod(token, case.dcp_interleave)
                local_group, owner = divmod(group, case.dcp_size)
                if owner == case.dcp_rank:
                    page, offset = divmod(local_group * case.dcp_interleave + lane, case.block_size)
                    if page < len(table):
                        value, valid = table[page] * case.block_size + offset, True
            row_values.append(value)
            row_valid.append(valid)
        values.append(row_values)
        validity.append(row_valid)
    return values, validity


def _cases(seed: int, count: int = CASES, **kwargs) -> list[tuple[int, Case]]:
    rng = random.Random(seed)
    return [(index, _random_case(rng, **kwargs)) for index in range(count)]


def test_oracle_matches_an_independent_restatement() -> None:
    for index, case in _cases(20261007):
        values, validity = _restated(case)
        out, counts = case.call()
        compact = case.compact_valid_to_front and case.dcp_size > 1
        for row, (row_out, count, row_values, row_valid) in enumerate(
            zip(out, counts, values, validity, strict=True)
        ):
            kept = [v for v, ok in zip(row_values, row_valid, strict=True) if ok]
            expected = (
                kept + [-1] * (len(row_values) - len(kept))
                if compact
                else [v if ok else -1 for v, ok in zip(row_values, row_valid, strict=True)]
            )
            assert (row_out, count) == (expected, len(kept)), f"case {index}, row {row}"


def test_compaction_is_a_stable_filter_of_the_column_layout() -> None:
    """Compacting never reorders, drops, or invents a valid mapping.

    Validity comes from the restatement, not from the values: negative page
    entries make legitimately mapped slots negative, even exactly ``-1``.
    """
    for index, case in _cases(7):
        if case.dcp_size == 1:
            case = replace(case, dcp_size=2, dcp_rank=case.dcp_rank % 2)
        _, validity = _restated(case)
        columns, column_counts = case.call(compact_valid_to_front=False)
        compacted, compact_counts = case.call(compact_valid_to_front=True)
        assert column_counts == compact_counts, f"case {index}"
        for row, (by_column, by_prefix, valid, count) in enumerate(
            zip(columns, compacted, validity, compact_counts, strict=True)
        ):
            kept = [value for value, ok in zip(by_column, valid, strict=True) if ok]
            assert by_prefix[:count] == kept, f"case {index}, row {row}"
            assert by_prefix[count:] == [-1] * (len(by_prefix) - count), f"case {index}"


def test_ranks_partition_the_owned_tokens() -> None:
    """Across all ranks, every in-range nonnegative token maps exactly once."""
    for index, case in _cases(11, nonnegative_pages=True):
        covered = case.block_size * len(case.block_table[0]) * case.dcp_size
        rows = [[t if t < covered else t % covered for t in row] for row in case.rows]
        layouts = [
            case.call(rows=rows, dcp_rank=rank, compact_valid_to_front=False)
            for rank in range(case.dcp_size)
        ]
        for row_index, row in enumerate(rows):
            owners = [
                [rank for rank, (out, _) in enumerate(layouts) if out[row_index][column] >= 0]
                for column in range(len(row))
            ]
            for column, token in enumerate(row):
                expected = 1 if token >= 0 else 0
                assert len(owners[column]) == expected, f"case {index}, row {row_index}"
            assert sum(counts[row_index] for _, counts in layouts) == sum(t >= 0 for t in row)


def test_single_rank_is_the_identity_paging_of_every_column() -> None:
    for index, case in _cases(13):
        case = replace(case, dcp_size=1, dcp_rank=0)
        for compact in (False, True):
            out, counts = case.call(compact_valid_to_front=compact)
            for row_index, (request, row) in enumerate(zip(case.req_ids, case.rows, strict=True)):
                table = case.block_table[request]
                expected = [
                    table[t // case.block_size] * case.block_size + t % case.block_size
                    if 0 <= t and t // case.block_size < len(table)
                    else -1
                    for t in row
                ]
                assert out[row_index] == expected, f"case {index}, compact={compact}"
                assert counts[row_index] == sum(
                    0 <= t and t // case.block_size < len(table) for t in row
                )


def test_rows_are_independent() -> None:
    for index, case in _cases(17, count=CASES // 2):
        out, counts = case.call()
        for row, (request, tokens) in enumerate(zip(case.req_ids, case.rows, strict=True)):
            alone = case.call(req_ids=[request], rows=[tokens])
            assert alone == ([out[row]], [counts[row]]), f"case {index}, row {row}"


def test_unreferenced_page_entries_do_not_affect_the_result() -> None:
    for index, case in _cases(19, count=CASES // 2):
        referenced = set()
        for request, row in zip(case.req_ids, case.rows, strict=True):
            for token in row:
                if token >= 0:
                    group, lane = divmod(token, case.dcp_interleave)
                    local_group, owner = divmod(group, case.dcp_size)
                    if owner == case.dcp_rank:
                        page = (local_group * case.dcp_interleave + lane) // case.block_size
                        referenced.add((request, page))
        perturbed = [
            [value if (request, page) in referenced else value + 7919 for page, value in enumerate(row)]
            for request, row in enumerate(case.block_table)
        ]
        assert case.call(block_table=perturbed) == case.call(), f"case {index}"


def test_relabeling_requests_with_their_table_rows_is_invisible() -> None:
    for index, case in _cases(23, count=CASES // 2):
        order = list(range(len(case.block_table)))
        random.Random(index).shuffle(order)
        position = {old: new for new, old in enumerate(order)}
        table = [case.block_table[old] for old in order]
        req_ids = [position[request] for request in case.req_ids]
        assert case.call(block_table=table, req_ids=req_ids) == case.call(), f"case {index}"


@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_deinterleave_is_a_bijection_onto_local_positions(interleave: int) -> None:
    """Each rank's owned global positions enumerate its local positions once."""
    dcp_size, block_size = 4, 8 * interleave
    span = dcp_size * block_size * 6
    table = [list(range(span // (dcp_size * block_size) + 1))]
    for rank in range(dcp_size):
        out, _ = localize_reference(
            [0], table, [list(range(span))],
            block_size=block_size, dcp_size=dcp_size, dcp_rank=rank,
            dcp_interleave=interleave, compact_valid_to_front=False,
        )
        # With an identity page table, physical slots equal local positions.
        local = [value for value in out[0] if value >= 0]
        assert local == list(range(span // dcp_size))
