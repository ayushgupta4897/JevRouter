# Vendored code

`vendor/switchyard/` is a source snapshot of [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard),
pinned at commit [`21e9e4a`](https://github.com/NVIDIA-NeMo/Switchyard/commit/21e9e4aefd6739269ba19633adb68df830e72c8f)
(2026-09-21), licensed Apache-2.0, Copyright NVIDIA Corporation. Its own `LICENSE` and `NOTICE`
files are kept intact inside that directory and govern that code.

It is vendored rather than fetched at install time for two reasons: Switchyard is pre-1.0 and its
Python API has moved ahead of the last PyPI release (the project's own README says to build from
source until the next one), and this repository should build and run offline from a single clone
without depending on GitHub being reachable at build time.

**Do not hand-edit files under `vendor/switchyard/`.** To pick up a newer Switchyard, re-run
`scripts/update_vendor.sh <commit-or-branch>`, review the diff, and re-run `scripts/build.sh` and
`scripts/e2e.sh` before committing. `switchyard_rust/*.so` build artifacts are intentionally
excluded (`.gitignore`); `scripts/build.sh` regenerates them.

Everything outside `vendor/` (`jevjudge/`, `switchyard/*.toml`, `scripts/`, `examples/`, `docs/`)
is this project's own code.
