# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hooks for repository-relative source links."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, unquote, urlsplit

_FENCE_RE = re.compile(
    r"^(?P<prefix>[> \t]*)(?:(?P<list_marker>[-+*]|\d{1,9}[.)])"
    r"(?P<list_padding>[ \t]{1,4}))?(?P<marker>`{3,}|~{3,})(?P<rest>[^\r\n]*)"
)
_BACKTICK_RUN_RE = re.compile(r"`+")
_LINK_DESTINATION_RE = re.compile(
    r"(?P<prefix>\]\(\s*)(?:<(?P<angled>[^>\r\n]+)>|(?P<plain>[^\s)\r\n]+))"
)


class _PageFile(Protocol):
    abs_src_path: str


class _Page(Protocol):
    file: _PageFile


class _MkDocsConfig(Protocol):
    config_file_path: str

    def __getitem__(self, key: str) -> object: ...


def _target_url(
    destination: str,
    *,
    source_path: Path,
    docs_dir: Path,
    repo_root: Path,
    repo_url: str,
    source_ref: str,
) -> str | None:
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith(("/", "\\")):
        return None

    try:
        target = (source_path.parent / unquote(parsed.path)).resolve()
    except (OSError, RuntimeError):
        return None

    if target.is_relative_to(docs_dir) or not target.is_relative_to(repo_root):
        return None
    if not target.exists():
        return None

    object_type = "tree" if target.is_dir() else "blob"
    relative_target = quote(target.relative_to(repo_root).as_posix(), safe="/")
    encoded_ref = quote(source_ref, safe="/")
    suffix = destination[len(parsed.path) :]
    return f"{repo_url.rstrip('/')}/{object_type}/{encoded_ref}/{relative_target}{suffix}"


def _rewrite_link_destinations(
    markdown: str,
    *,
    source_path: Path,
    docs_dir: Path,
    repo_root: Path,
    repo_url: str,
    source_ref: str,
) -> str:
    def replace(match: re.Match[str]) -> str:
        destination = match.group("angled") or match.group("plain")
        rewritten = _target_url(
            destination,
            source_path=source_path,
            docs_dir=docs_dir,
            repo_root=repo_root,
            repo_url=repo_url,
            source_ref=source_ref,
        )
        if rewritten is None:
            return match.group(0)
        if match.group("angled") is not None:
            rewritten = f"<{rewritten}>"
        return f"{match.group('prefix')}{rewritten}"

    return _LINK_DESTINATION_RE.sub(replace, markdown)


def _rewrite_outside_code_spans(
    markdown: str,
    *,
    source_path: Path,
    docs_dir: Path,
    repo_root: Path,
    repo_url: str,
    source_ref: str,
) -> str:
    def rewrite(text: str) -> str:
        return _rewrite_link_destinations(
            text,
            source_path=source_path,
            docs_dir=docs_dir,
            repo_root=repo_root,
            repo_url=repo_url,
            source_ref=source_ref,
        )

    output: list[str] = []
    cursor = 0
    while opening := _BACKTICK_RUN_RE.search(markdown, cursor):
        output.append(rewrite(markdown[cursor : opening.start()]))
        closing = next(
            (
                candidate
                for candidate in _BACKTICK_RUN_RE.finditer(markdown, opening.end())
                if len(candidate.group(0)) == len(opening.group(0))
            ),
            None,
        )
        if closing is None:
            output.append(rewrite(markdown[opening.start() :]))
            return "".join(output)
        output.append(markdown[opening.start() : closing.end()])
        cursor = closing.end()
    output.append(rewrite(markdown[cursor:]))
    return "".join(output)


def _container_position(line: str, quote_depth: int) -> tuple[int, int] | None:
    cursor = 0
    for _ in range(quote_depth):
        padding = 0
        while cursor < len(line) and line[cursor] == " " and padding < 3:
            cursor += 1
            padding += 1
        if cursor >= len(line) or line[cursor] != ">":
            return None
        cursor += 1
        if cursor < len(line) and line[cursor] in " \t":
            cursor += 1

    indentation_start = cursor
    while cursor < len(line) and line[cursor] in " \t":
        cursor += 1
    indentation = len(line[indentation_start:cursor].expandtabs(4))
    return indentation, cursor


def _fence_container(match: re.Match[str]) -> tuple[int, int] | None:
    prefix = match.group("prefix")
    quote_depth = prefix.count(">")
    position = _container_position(prefix, quote_depth)
    if position is None:
        return None

    indentation, _ = position
    list_marker = match.group("list_marker")
    if list_marker is not None:
        indentation += len(list_marker) + len(match.group("list_padding").expandtabs(4))
    elif indentation <= 3:
        indentation = 0
    return quote_depth, indentation


def _line_is_in_container(line: str, quote_depth: int, indentation: int) -> bool:
    if not line.strip():
        return True
    position = _container_position(line, quote_depth)
    if position is None:
        return False
    line_indentation, content_start = position
    return not line[content_start:].strip() or line_indentation >= indentation


def _is_closing_fence(
    match: re.Match[str],
    *,
    fence_character: str,
    fence_length: int,
    quote_depth: int,
    indentation: int,
) -> bool:
    marker = match.group("marker")
    if (
        match.group("list_marker") is not None
        or marker[0] != fence_character
        or len(marker) < fence_length
        or match.group("rest").strip()
        or match.group("prefix").count(">") != quote_depth
    ):
        return False

    position = _container_position(match.group("prefix"), quote_depth)
    if position is None:
        return False
    closing_indentation, _ = position
    return closing_indentation <= 3 if indentation == 0 else closing_indentation == indentation


def rewrite_repository_links(
    markdown: str,
    *,
    source_path: Path,
    docs_dir: Path,
    repo_root: Path,
    repo_url: str,
    source_ref: str,
) -> str:
    """Rewrite valid links outside docs_dir to repository source URLs."""
    source_path = source_path.resolve()
    docs_dir = docs_dir.resolve()
    repo_root = repo_root.resolve()
    output: list[str] = []
    pending_markdown: list[str] = []
    fence_lines: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    fence_quote_depth = 0
    fence_indentation = 0

    def flush_pending_markdown() -> None:
        if not pending_markdown:
            return
        output.append(
            _rewrite_outside_code_spans(
                "".join(pending_markdown),
                source_path=source_path,
                docs_dir=docs_dir,
                repo_root=repo_root,
                repo_url=repo_url,
                source_ref=source_ref,
            )
        )
        pending_markdown.clear()

    for line in markdown.splitlines(keepends=True):
        if fence_character is not None and not _line_is_in_container(
            line, fence_quote_depth, fence_indentation
        ):
            pending_markdown.extend(fence_lines)
            fence_lines.clear()
            fence_character = None
            fence_length = 0
            fence_quote_depth = 0
            fence_indentation = 0

        fence = _FENCE_RE.match(line)
        if fence_character is None:
            container = _fence_container(fence) if fence is not None else None
            if fence is None or container is None:
                pending_markdown.append(line)
                continue
            flush_pending_markdown()
            marker = fence.group("marker")
            fence_character = marker[0]
            fence_length = len(marker)
            fence_quote_depth, fence_indentation = container
            fence_lines.append(line)
            continue

        fence_lines.append(line)
        if fence is not None and _is_closing_fence(
            fence,
            fence_character=fence_character,
            fence_length=fence_length,
            quote_depth=fence_quote_depth,
            indentation=fence_indentation,
        ):
            output.extend(fence_lines)
            fence_lines.clear()
            fence_character = None
            fence_length = 0
            fence_quote_depth = 0
            fence_indentation = 0

    output.extend(fence_lines)
    flush_pending_markdown()
    return "".join(output)


def on_page_markdown(
    markdown: str,
    *,
    page: _Page,
    config: _MkDocsConfig,
    files: object,
) -> str:
    """Rewrite repository-relative links before MkDocs validates Markdown links."""
    del files
    docs_dir = config["docs_dir"]
    repo_url = config["repo_url"]
    extra = config["extra"]
    if not isinstance(docs_dir, str) or not isinstance(repo_url, str):
        raise ValueError("MkDocs docs_dir and repo_url must be configured")
    if not isinstance(extra, Mapping) or not isinstance(extra.get("source_ref"), str):
        raise ValueError("MkDocs extra.source_ref must be configured")

    return rewrite_repository_links(
        markdown,
        source_path=Path(page.file.abs_src_path),
        docs_dir=Path(docs_dir),
        repo_root=Path(config.config_file_path).resolve().parent,
        repo_url=repo_url,
        source_ref=extra["source_ref"],
    )
