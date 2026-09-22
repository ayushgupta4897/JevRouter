# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for repository-relative links in MkDocs pages."""

import re
from pathlib import Path
from types import SimpleNamespace

from mkdocs_hooks import on_page_markdown, rewrite_repository_links

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/example/switchyard"
SOURCE_REF = "docs-ref"


class _Config(dict[str, object]):
    def __init__(self, repo_root: Path, docs_dir: Path) -> None:
        super().__init__(
            docs_dir=str(docs_dir),
            repo_url=REPO_URL,
            extra={"source_ref": SOURCE_REF},
        )
        self.config_file_path = str(repo_root / "mkdocs.yml")


def _repository(
    tmp_path: Path, page_relative_path: str = "docs/guides/page.md"
) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    docs_dir = repo_root / "docs"
    page_path = repo_root / page_relative_path
    page_path.parent.mkdir(parents=True)
    page_path.write_text("# Page\n", encoding="utf-8")
    (repo_root / "AGENTS.md").write_text("# Instructions\n", encoding="utf-8")
    (repo_root / "examples" / "prometheus").mkdir(parents=True)
    (docs_dir / "index.md").write_text("# Home\n", encoding="utf-8")
    return repo_root, docs_dir, page_path


def _rewrite(markdown: str, repo_root: Path, docs_dir: Path, page_path: Path) -> str:
    return rewrite_repository_links(
        markdown,
        source_path=page_path,
        docs_dir=docs_dir,
        repo_root=repo_root,
        repo_url=REPO_URL,
        source_ref=SOURCE_REF,
    )


def test_repository_file_and_directory_links_use_blob_and_tree_urls(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    markdown = "[file](../../AGENTS.md)\n[directory](../../examples/prometheus/)\n"

    rewritten = _rewrite(markdown, repo_root, docs_dir, page_path)

    assert rewritten == (
        f"[file]({REPO_URL}/blob/{SOURCE_REF}/AGENTS.md)\n"
        f"[directory]({REPO_URL}/tree/{SOURCE_REF}/examples/prometheus)\n"
    )


def test_hook_resolves_links_from_a_nested_page_and_uses_mkdocs_config(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path, "docs/routing/deep/page.md")
    page = SimpleNamespace(file=SimpleNamespace(abs_src_path=str(page_path)))

    rewritten = on_page_markdown(
        "[instructions](../../../AGENTS.md)",
        page=page,
        config=_Config(repo_root, docs_dir),
        files=object(),
    )

    assert rewritten == f"[instructions]({REPO_URL}/blob/{SOURCE_REF}/AGENTS.md)"


def test_repository_links_preserve_query_strings_and_fragments(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)

    rewritten = _rewrite(
        "[instructions](../../AGENTS.md?plain=1&view=raw#setup)",
        repo_root,
        docs_dir,
        page_path,
    )

    assert rewritten == (
        f"[instructions]({REPO_URL}/blob/{SOURCE_REF}/AGENTS.md?plain=1&view=raw#setup)"
    )


def test_repository_links_in_fenced_code_blocks_are_not_rewritten(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    markdown = (
        "```markdown\n"
        "[backtick fence](../../AGENTS.md)\n"
        "```\n"
        "~~~markdown\n"
        "[tilde fence](../../AGENTS.md)\n"
        "~~~\n"
        "[live link](../../AGENTS.md)\n"
    )

    rewritten = _rewrite(markdown, repo_root, docs_dir, page_path)

    assert rewritten == markdown.replace(
        "[live link](../../AGENTS.md)",
        f"[live link]({REPO_URL}/blob/{SOURCE_REF}/AGENTS.md)",
    )


def test_repository_links_in_blockquote_fences_are_not_rewritten(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    markdown = (
        "> ~~~markdown\n"
        "> [fenced link](../../AGENTS.md)\n"
        "> ~~~\n"
        "> Ordinary prose with ~~~ inline.\n"
        "> [live link](../../AGENTS.md)\n"
    )

    rewritten = _rewrite(markdown, repo_root, docs_dir, page_path)

    assert rewritten == markdown.replace(
        "[live link](../../AGENTS.md)",
        f"[live link]({REPO_URL}/blob/{SOURCE_REF}/AGENTS.md)",
    )


def test_repository_links_in_list_item_fences_are_not_rewritten(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    markdown = (
        "- Example:\n"
        "\n"
        "    ~~~markdown\n"
        "    [fenced link](../../AGENTS.md)\n"
        "    ~~~\n"
        "\n"
        "    Ordinary prose with ~~~ inline.\n"
        "    [live link](../../AGENTS.md)\n"
    )

    rewritten = _rewrite(markdown, repo_root, docs_dir, page_path)

    assert rewritten == markdown.replace(
        "[live link](../../AGENTS.md)",
        f"[live link]({REPO_URL}/blob/{SOURCE_REF}/AGENTS.md)",
    )


def test_indented_fence_markers_do_not_hide_outdented_prose_links(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    markdown = (
        "    ~~~\n"
        "[live link](../../AGENTS.md)\n"
        "    ~~~\n"
    )

    rewritten = _rewrite(markdown, repo_root, docs_dir, page_path)

    assert rewritten == markdown.replace(
        "[live link](../../AGENTS.md)",
        f"[live link]({REPO_URL}/blob/{SOURCE_REF}/AGENTS.md)",
    )


def test_missing_repository_target_remains_relative(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    markdown = "[missing](../../missing.md?plain=1#section)"

    assert _rewrite(markdown, repo_root, docs_dir, page_path) == markdown


def test_existing_target_outside_repository_remains_relative(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    (tmp_path / "outside.md").write_text("# Outside\n", encoding="utf-8")
    markdown = "[outside](../../../outside.md)"

    assert _rewrite(markdown, repo_root, docs_dir, page_path) == markdown


def test_links_inside_docs_dir_and_non_relative_links_are_unchanged(tmp_path: Path) -> None:
    repo_root, docs_dir, page_path = _repository(tmp_path)
    markdown = (
        "[docs page](../index.md)\n"
        "[external](https://example.com/docs)\n"
        "[page anchor](#section)\n"
    )

    assert _rewrite(markdown, repo_root, docs_dir, page_path) == markdown


def test_docs_do_not_hard_code_same_repository_blob_or_tree_links() -> None:
    config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    repo_url_match = re.search(r"^repo_url:\s*(\S+)\s*$", config, flags=re.MULTILINE)
    assert repo_url_match is not None
    repo_url = repo_url_match.group(1).strip("'\"").rstrip("/")
    source_link = re.compile(rf"{re.escape(repo_url)}/(?:blob|tree)/")
    markdown_paths = [REPO_ROOT / "README.md", *(REPO_ROOT / "docs").rglob("*.md")]

    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}"
        for path in markdown_paths
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if source_link.search(line)
    ]

    assert not offenders, "Hard-coded same-repository source links:\n" + "\n".join(offenders)
