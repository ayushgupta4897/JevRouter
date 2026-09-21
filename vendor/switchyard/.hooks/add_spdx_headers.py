#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add or upgrade SPDX headers on Python files.

Header form (per NVIDIA OSRB policy + SPDX Annex E):

    # SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0

Behaviour:
- If the file already carries the exact required header: no change.
- If the file carries the legacy short-form header
  (``# SPDX-License-Identifier: Apache-2.0`` + ``# Copyright (c) NVIDIA Corporation``,
  in either order, contiguous): the two lines are replaced in place.
- Otherwise: the header is prepended (preserving a leading shebang).
"""
import re
import sys
from pathlib import Path

COPYRIGHT_YEAR = "2026"
COPYRIGHT_LINE = (
    f"# SPDX-FileCopyrightText: Copyright (c) {COPYRIGHT_YEAR} "
    "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0"
SPDX_HEADER = f"{COPYRIGHT_LINE}\n{LICENSE_LINE}\n"

# Matches the legacy two-line short form in either order, optionally with a
# blank line between (we don't expect one, but be defensive). Anchored at the
# start of file or directly after a shebang line — see _split_shebang.
_LEGACY_RE = re.compile(
    r"(?:# SPDX-License-Identifier:\s*Apache-2\.0\n# Copyright \(c\) NVIDIA Corporation\n"
    r"|# Copyright \(c\) NVIDIA Corporation\n# SPDX-License-Identifier:\s*Apache-2\.0\n)"
)


def _split_shebang(content: str) -> tuple[str, str]:
    """Return (shebang_with_newline, rest). Empty shebang if none."""
    if content.startswith("#!"):
        nl = content.find("\n")
        if nl == -1:
            return content + "\n", ""
        return content[: nl + 1], content[nl + 1 :]
    return "", content


def has_required_header(filepath: Path) -> bool:
    """True iff the file's header already matches the required form exactly."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read(512)
    except OSError:
        return False
    _, body = _split_shebang(content)
    return body.startswith(SPDX_HEADER)


def update_spdx_header(filepath: Path) -> bool:
    """Ensure the required header is present. Return True if the file was modified."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return False

    if not content.strip():
        return False

    shebang, body = _split_shebang(content)

    if body.startswith(SPDX_HEADER):
        return False

    legacy_match = _LEGACY_RE.match(body)
    if legacy_match:
        new_body = SPDX_HEADER + body[legacy_match.end() :]
    else:
        # Fresh file — prepend, ensure a blank line before existing content.
        if body.startswith("\n"):
            new_body = SPDX_HEADER + body
        else:
            new_body = SPDX_HEADER + "\n" + body

    new_content = shebang + new_body
    if new_content == content:
        return False

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        print(f"Error writing {filepath}: {e}", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        files = [Path(f) for f in argv if f.endswith(".py")]
    else:
        files = list(Path(".").rglob("*.py"))

    skip_parts = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", ".eggs"}

    count = 0
    for filepath in files:
        if any(part in filepath.parts for part in skip_parts):
            continue
        if filepath.is_file() and update_spdx_header(filepath):
            count += 1
            print(f"Updated SPDX header: {filepath}")

    if not argv:
        print(f"\nTotal files updated: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
