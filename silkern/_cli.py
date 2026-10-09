"""Command-line plumbing shared by the verification entry points."""

from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn

#: Exit status of a usage error, BSD's ``EX_USAGE``. The verifiers reserve 2 for
#: "backend unavailable" under ``--require-device``, which argparse would also
#: use for a mistyped option, so a CI step could not tell the two apart.
USAGE_ERROR = 64


def program_name(module: str) -> str:
    """The name a user typed: the console script, or ``python -m module``.

    Under ``python -m`` argparse would print the file name, such as
    ``__main__.py`` or ``mlx_verify.py``, which cannot be typed back.
    """
    invoked = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if not invoked or invoked.endswith(".py") or invoked == "-c":
        return f"python -m {module}"
    return invoked


class ArgumentParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` whose usage errors exit :data:`USAGE_ERROR`."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(USAGE_ERROR, f"{self.prog}: error: {message}\n")
