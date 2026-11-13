"""Command-line entry point: ``python -m evomorph <command> ...``.

Subcommands are discovered from ``evomorph/commands/*.py``: every module whose
name does not start with ``_`` must expose ``register(subparsers)``, which adds
its subparser and sets ``func`` (a callable taking the parsed args and returning
an exit code). Adding a command never requires editing this file.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

import evomorph.commands as commands_pkg
from evomorph.commands._common import GenomeLoadError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evomorph", description="EVOMORPH evolutionary robotics laboratory"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for info in sorted(pkgutil.iter_modules(commands_pkg.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{commands_pkg.__name__}.{info.name}")
        register = getattr(module, "register", None)
        if callable(register):
            register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except GenomeLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
