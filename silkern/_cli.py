"""Command-line plumbing shared by the verification entry points."""

from __future__ import annotations

import argparse
import sys

#: Exit status of a usage error, BSD's ``EX_USAGE``. The verifiers reserve 2 for
#: "backend unavailable" under ``--require-device``, which argparse would also
#: use for a mistyped option, so a CI step could not tell the two apart.
USAGE_ERROR = 64


class ArgumentParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` whose usage errors exit :data:`USAGE_ERROR`."""

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(USAGE_ERROR, f"{self.prog}: error: {message}\n")
