"""Deterministic sparse-index localization for Python, CUDA, and Apple MLX.

``localize_reference`` defines the contract in dependency-free Python.
``localize_rowwise`` and ``localize_hierarchical`` use caller-owned CUDA
buffers; ``localize_mlx`` returns new lazy MLX arrays on Apple silicon.
All preserve input-relative survivor order and exact valid counts.

Use ``conformance()`` for CUDA or ``conformance_mlx()`` for Apple validation.
See ``docs/architecture.md`` for backend-specific memory and execution models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from silkern.contract import (
    DEFAULT_TILE_SIZE,
    MAX_ROW_WIDTH,
    SUPPORTED_TILE_SIZES,
    localize_reference,
)
from silkern.errors import LocalizationError
from silkern.kernels import localize_hierarchical, localize_rowwise
from silkern.mlx import localize_mlx
from silkern.verify import CellReport, ConformanceReport, conformance
from silkern.workspace import workspace_shapes

if TYPE_CHECKING:
    from silkern.mlx_verify import MLXConformanceReport


__version__ = "2.0.0"


def conformance_mlx(
    *, backend: str = "both", matrix: Sequence[dict[str, int | bool]] | None = None,
    batch: int = 8, seed: int = 0, repeats: int = 3,
) -> MLXConformanceReport:
    """Verify Apple backends; see :func:`silkern.mlx_verify.conformance_mlx`."""
    from silkern.mlx_verify import conformance_mlx as run

    return run(backend=backend, matrix=matrix, batch=batch, seed=seed, repeats=repeats)


__all__ = [
    "DEFAULT_TILE_SIZE",
    "MAX_ROW_WIDTH",
    "SUPPORTED_TILE_SIZES",
    "CellReport",
    "ConformanceReport",
    "LocalizationError",
    "conformance",
    "conformance_mlx",
    "localize_hierarchical",
    "localize_mlx",
    "localize_reference",
    "localize_rowwise",
    "workspace_shapes",
]
