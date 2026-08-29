"""Put the repository root on ``sys.path`` for the whole test session.

``silkern`` is installed (``pip install -e .``), but ``bench/`` deliberately is
not -- it holds diagnostics, not library code, and is excluded from the wheel.
The tests still import it to keep it from rotting, and a root ``conftest.py`` is
what makes that work under a bare ``pytest`` as well as ``python -m pytest``.
"""
