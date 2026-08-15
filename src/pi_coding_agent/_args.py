"""TS ``cli/args.ts`` compatible parseArgs/printHelp helpers.

The canonical CLI still uses argparse internally; these functions expose the
same conceptual result as the TS parser for programmatic callers.
"""

from __future__ import annotations

from typing import Any


def parseArgs(args: list[str] | None = None):
    """Parse CLI arguments TS-style.

    Long unknown flags are preserved in ``unknown_flags`` (argparse short
    unknown options are left in ``diagnostics``/``unknown``).
    """
    from ._cli import _create_parser

    parser = _create_parser()
    parsed, unknown = parser.parse_known_args(args)
    file_args: list[str] = []
    messages: list[str] = []
    for part in parsed.message:
        if part.startswith("@"):
            file_args.append(part[1:])
        else:
            messages.append(part)
    parsed.messages = messages
    parsed.message = messages
    parsed.file_args = file_args
    parsed.unknown = list(unknown)
    parsed.unknown_flags = _unknown_flags_from_tokens(unknown)
    parsed.diagnostics = []
    return parsed


def _unknown_flags_from_tokens(tokens: list[str]) -> dict[str, bool | str]:
    result: dict[str, bool | str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            name, value = token[2:].split("=", 1)
            result[name] = value
        else:
            name = token[2:]
            next_index = index + 1
            if next_index < len(tokens) and not tokens[next_index].startswith("-"):
                result[name] = tokens[next_index]
                index = next_index
            else:
                result[name] = True
        index += 1
    return result


def printHelp(extension_flags: list[Any] | None = None) -> None:
    """Print CLI help, optionally followed by extension flags."""
    from ._cli import _create_parser

    parser = _create_parser()
    print(parser.format_help().rstrip())
    if extension_flags:
        print("\nExtension CLI Flags:")
        for flag in extension_flags:
            option = f"--{flag.name}"
            value = " <value>" if getattr(flag, "type", "boolean") != "boolean" else ""
            description = getattr(flag, "description", None) or "Extension flag"
            print(f"  {option}{value:<24} {description}")


__all__ = ["parseArgs", "printHelp"]
