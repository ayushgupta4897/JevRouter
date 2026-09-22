"""``routerctl validate|compile|serve`` -- see routerctl/README.md."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .compiler import CompileResult, ConfigError, compile_directory


def _find_dry_run_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    repo_default = Path(__file__).resolve().parents[3] / "vendor" / "switchyard" / "target" / "release" / "switchyard-server"
    if repo_default.exists():
        return str(repo_default)
    found = shutil.which("switchyard-server")
    if found:
        return found
    raise SystemExit(
        "switchyard-server not found. Run scripts/build.sh, or pass --switchyard-server /path/to/binary."
    )


def _dry_run(binary: str, result: CompileResult) -> tuple[bool, str]:
    # --dry-run makes no network calls, but Switchyard still requires every referenced
    # api_key_env to exist and be non-empty. Fill in a placeholder for any not already set in
    # this environment, so `routerctl validate` works in CI or a laptop with no real secrets --
    # `routerctl serve` (supervisor.py) does NOT do this: a real launch needs a real key.
    env = {**os.environ, **{name: os.environ.get(name) or "routerctl-dry-run-placeholder" for name in result.api_key_envs}}
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
        handle.write(result.toml)
        path = handle.name
    try:
        proc = subprocess.run([binary, "--config", path, "--dry-run"], capture_output=True, text=True, timeout=30, env=env)
        ok = proc.returncode == 0
        output = (proc.stdout + proc.stderr).strip()
        return ok, output
    finally:
        Path(path).unlink(missing_ok=True)


def cmd_validate(args: argparse.Namespace) -> int:
    teams_dir = Path(args.teams_dir)
    try:
        result = compile_directory(teams_dir, Path(args.clients) if args.clients else None)
    except ConfigError as error:
        print(f"INVALID  {error.file or teams_dir}: {error}", file=sys.stderr)
        return 1
    print(f"schema OK: {len(result.route_names)} route(s) across "
          f"{len({s.team for s in result.route_names.values()})} team(s)")
    for name, source in sorted(result.route_names.items()):
        print(f"  {name}  <- {source.file} (team: {source.team})")

    if args.skip_dry_run:
        return 0
    binary = _find_dry_run_binary(args.switchyard_server)
    ok, output = _dry_run(binary, result)
    if not ok:
        print(f"\nswitchyard-server --dry-run FAILED:\n{output}", file=sys.stderr)
        return 1
    print(f"\nswitchyard-server --dry-run OK: {output}")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    teams_dir = Path(args.teams_dir)
    try:
        result = compile_directory(teams_dir, Path(args.clients) if args.clients else None)
    except ConfigError as error:
        print(f"INVALID  {error.file or teams_dir}: {error}", file=sys.stderr)
        return 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.toml)
    print(f"wrote {out_path} ({len(result.route_names)} routes)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .supervisor import run_supervisor  # deferred: pulls in fastapi/uvicorn/httpx

    return run_supervisor(
        teams_dir=Path(args.teams_dir),
        clients_path=Path(args.clients) if args.clients else None,
        public_port=args.port,
        switchyard_server=_find_dry_run_binary(args.switchyard_server),
        poll_seconds=args.poll_seconds,
        build_dir=Path(args.build_dir),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routerctl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("teams_dir", help="Directory of per-team *.yaml route files.")
    common.add_argument("--clients", help="Path to clients.yaml (default: <teams_dir>/../clients.yaml).")
    common.add_argument("--switchyard-server", help="Path to the switchyard-server binary.")

    p_validate = sub.add_parser("validate", parents=[common], help="Schema-check every team file, then switchyard-server --dry-run the compiled result.")
    p_validate.add_argument("--skip-dry-run", action="store_true", help="Schema-check only; skip the Switchyard binary check.")
    p_validate.set_defaults(func=cmd_validate)

    p_compile = sub.add_parser("compile", parents=[common], help="Compile to a single TOML file.")
    p_compile.add_argument("-o", "--output", default="build/router.toml")
    p_compile.set_defaults(func=cmd_compile)

    p_serve = sub.add_parser("serve", parents=[common], help="Serve with live reload: watch, recompile, validate, health-check, then swap.")
    p_serve.add_argument("--port", type=int, default=4000, help="Public port clients connect to.")
    p_serve.add_argument("--poll-seconds", type=float, default=1.5)
    p_serve.add_argument("--build-dir", default="build")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
