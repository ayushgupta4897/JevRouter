"""Compile a platform ``clients.yaml`` plus a set of team YAML files into one Switchyard TOML.

Each of the four policies compiles to the exact TOML shape validated end to end against real
Switchyard and real Jev elsewhere in this repository (see `switchyard/routes.*.toml`,
`docs/DECISION.md` section 9). This module only assembles that shape from friendlier input; it
does not change what Switchyard does with it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .schema import (
    JEV_JUDGE_CLIENT,
    JEV_JUDGE_MODEL,
    AutoPolicy,
    ClientsFile,
    ComplexityPolicy,
    ConfigError,
    EscalationPolicy,
    IntentPolicy,
    ModelRef,
    Route,
    TeamFile,
)

DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:8090/v1"
_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


@dataclass
class Source:
    """Where one piece of compiled TOML came from, for error messages and the header comment."""

    file: str
    team: str | None = None


@dataclass
class CompileResult:
    toml: str
    route_names: dict[str, Source] = field(default_factory=dict)  # route name -> its source
    api_key_envs: set[str] = field(default_factory=set)  # every api_key_env a client references


def _table_key(route_name: str) -> str:
    """A TOML-safe table key derived from a route's client-visible name (which keeps '/')."""
    return _KEY_UNSAFE.sub("_", route_name).strip("_")


def _toml_str(value: str) -> str:
    return json.dumps(value)  # TOML basic strings are JSON-string-compatible for our alphabet


def _toml_inline(value: object) -> str:
    """A TOML inline value: JSON's scalar and dict syntax coincide with TOML's closely enough
    for the plain scalars and one level of nesting `extra_body` actually needs (e.g.
    `{"reasoning": {"effort": "high"}}`); booleans and numbers round-trip as-is via json.dumps."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml_inline(v)}" for k, v in value.items()) + " }"
    return json.dumps(value)


class TargetRegistry:
    """Emits one `[targets.*]` block per unique (model id, client) pair, however many routes
    reference it.

    Switchyard itself dedups targets this way -- two target tables with the same id and
    llm_client collapse into one physical target at runtime, silently keeping one and dropping
    the other (a `--dry-run` warning is the only signal, and which one survives is unspecified).
    Matching that key here turns a same-model-different-settings collision into a clear
    compile-time error naming both routes, instead of a runtime warning and a coin flip.
    """

    def __init__(self) -> None:
        self._by_pair: dict[tuple[str, str], tuple[str, str | None, Source]] = {}
        self._used_keys: set[str] = set()
        self._blocks: list[str] = []  # rendered `[targets.*]` blocks, in first-seen order

    def resolve(self, model: ModelRef, source: Source, route_name: str) -> str:
        pair = (model.id, model.client)
        extra_json = json.dumps(model.extra_body, sort_keys=True) if model.extra_body else None
        existing = self._by_pair.get(pair)
        if existing is not None:
            key, existing_extra, existing_source = existing
            if existing_extra != extra_json:
                raise ConfigError(
                    f"route {route_name!r} in {source.file!r} uses model {model.id!r} on client "
                    f"{model.client!r} with different extra_body settings than the same "
                    f"model+client pair already used by a route in {existing_source.file!r}. "
                    "Switchyard treats identical (id, client) as one target regardless of "
                    "extra_body, so this would be decided unpredictably at runtime. Match the "
                    "settings, or give one of them a distinct client entry in clients.yaml.",
                    file=source.file,
                )
            return key
        base = f"{model.client}_{_KEY_UNSAFE.sub('_', model.id).strip('_')}"
        key = base
        suffix = 2
        while key in self._used_keys:
            key = f"{base}_{suffix}"
            suffix += 1
        self._used_keys.add(key)
        self._by_pair[pair] = (key, extra_json, source)

        block = [f"[targets.{key}]", f"id = {_toml_str(model.id)}", f"llm_client = {_toml_str(model.client)}"]
        if model.extra_body:
            for field_name, value in model.extra_body.items():
                block.append(f"extra_body.{field_name} = {_toml_inline(value)}")
        self._blocks.append("\n".join(block))
        return key

    def render(self) -> str:
        return "\n\n".join(self._blocks)


def load_clients_file(path: Path) -> ClientsFile:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML: {error}", file=str(path)) from error
    try:
        return ClientsFile.model_validate(raw)
    except Exception as error:  # pydantic ValidationError or our own ConfigError
        raise ConfigError(str(error), file=str(path)) from error


def load_team_file(path: Path) -> TeamFile:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML: {error}", file=str(path)) from error
    try:
        return TeamFile.model_validate(raw)
    except Exception as error:
        raise ConfigError(str(error), file=str(path)) from error


def _compile_route(route: Route, registry: TargetRegistry, source: Source) -> str:
    """Return the route's own `[routes.*]` block. Targets it needs are resolved (and, the first
    time, emitted) through `registry` as a side effect, not returned here."""
    policy = route.policy
    key = _table_key(route.name)
    resolve = lambda model: registry.resolve(model, source, route.name)  # noqa: E731
    out: list[str] = []

    if isinstance(policy, AutoPolicy):
        efficient = resolve(policy.models["efficient"])
        capable = resolve(policy.models["capable"])
        out.append(f"[routes.{key}]")
        out.append(f"id = {_toml_str(route.name)}")
        out.append('type = "auto"')
        out.append(f"capable_target = {_toml_str(capable)}")
        out.append(f"efficient_target = {_toml_str(efficient)}")

    elif isinstance(policy, ComplexityPolicy):
        weak = resolve(policy.models["weak"])
        strong = resolve(policy.models["strong"])
        judge = resolve(ModelRef(id=policy.judge.id, client=policy.judge.client))
        out.append(f"[routes.{key}]")
        out.append(f"id = {_toml_str(route.name)}")
        out.append('type = "llm_classifier"')
        out.append('mode = "capability"')
        out.append(f"classifier_target = {_toml_str(judge)}")
        out.append(f"strong_target = {_toml_str(strong)}")
        out.append(f"weak_target = {_toml_str(weak)}")
        out.append(f"base_threshold = {policy.base_threshold}")
        out.append(f"threshold_step = {policy.threshold_step}")
        out.append('classify_trigger = "user_turn"')

    elif isinstance(policy, EscalationPolicy):
        weak = resolve(policy.models["weak"])
        strong = resolve(policy.models["strong"])
        judge = resolve(ModelRef(id=policy.judge.id, client=policy.judge.client))
        out.append(f"[routes.{key}]")
        out.append(f"id = {_toml_str(route.name)}")
        out.append('type = "llm_classifier"')
        out.append('mode = "escalation"')
        out.append(f"classifier_target = {_toml_str(judge)}")
        out.append(f"strong_target = {_toml_str(strong)}")
        out.append(f"weak_target = {_toml_str(weak)}")
        out.append(
            f"escalation = {{ confirmations = {policy.confirmations}, "
            f"recent_turn_window = {policy.recent_turn_window}, window_message_chars = 500 }}"
        )

    elif isinstance(policy, IntentPolicy):
        bucket_names = list(policy.models)
        bucket_targets = {bucket: resolve(model) for bucket, model in policy.models.items()}
        judge = resolve(ModelRef(id=policy.judge.id, client=policy.judge.client))

        prompt_lines = ["Choose the model group that best fits this request."]
        for bucket, model in policy.models.items():
            description = " ".join((model.description or "").split())  # collapse to one line
            prompt_lines.append(f"- {bucket}: {description}")
        prompt_lines.append("Return JSON matching the response schema supplied with the request.")
        prompt = "\n".join(prompt_lines)

        response_schema = {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "object",
                    "properties": {"target": {"type": "string", "enum": bucket_names}},
                    "required": ["target"],
                    "additionalProperties": False,
                }
            },
            "required": ["decision"],
            "additionalProperties": False,
        }

        out.append(f"[routes.{key}]")
        out.append(f"id = {_toml_str(route.name)}")
        out.append('type = "llm_classifier"')
        out.append('mode = "custom"')
        out.append(f"default_target = {_toml_str(policy.default)}")
        out.append('classify_trigger = "user_turn"')
        out.append(f'prompt = """\n{prompt}\n"""')
        out.append(f"response_schema = '''\n{json.dumps(response_schema, indent=2)}\n'''")
        out.append("")
        out.append(f"[routes.{key}.models]")
        out.append(f"judge = [{_toml_str(judge)}]")
        for bucket in bucket_names:
            out.append(f"{bucket} = [{_toml_str(bucket_targets[bucket])}]")
        all_targets = ", ".join(_toml_str(bucket_targets[b]) for b in bucket_names)
        out.append(f"any = [{all_targets}]")
        out.append("")
        out.append(f"[routes.{key}.policy]")
        out.append('type = "target_selector"')
        out.append('selector = "/decision/target"')

    else:
        raise AssertionError(f"unhandled policy type: {type(policy).__name__}")  # pragma: no cover

    return "\n".join(out)


def compile_routes(clients: ClientsFile, teams: list[tuple[str, TeamFile]]) -> CompileResult:
    """``teams`` is ``[(filename, TeamFile), ...]`` -- filenames are only for error attribution."""
    header = ["# Generated by routerctl. Do not hand-edit; edit the YAML and recompile.", "schema_version = 1"]

    client_blocks: list[str] = []
    api_key_envs: set[str] = set()
    for client_name, client in clients.clients.items():
        block = [f"[llm_clients.{client_name}]", f"format = {_toml_str(client.format)}", f"base_url = {_toml_str(client.base_url)}"]
        if client.api_key_env:
            block.append(f"api_key_env = {_toml_str(client.api_key_env)}")
            api_key_envs.add(client.api_key_env)
        if client.timeout_ms is not None:
            block.append(f"timeout_ms = {client.timeout_ms}")
        if client.max_retries is not None:
            block.append(f"max_retries = {client.max_retries}")
        client_blocks.append("\n".join(block))

    # The shared judge client always exists, whether or not any team's `clients.yaml` mentions
    # it -- teams reference it by name (schema.JEV_JUDGE_CLIENT) without needing to declare it.
    client_blocks.append(
        "\n".join([f"[llm_clients.{JEV_JUDGE_CLIENT}]", 'format = "openai_chat"',
                   f"base_url = {_toml_str(DEFAULT_JUDGE_BASE_URL)}", "timeout_ms = 5000", "max_retries = 1"])
    )

    registry = TargetRegistry()
    route_names: dict[str, Source] = {}
    route_blocks: list[str] = []

    for filename, team_file in teams:
        for route in team_file.routes:
            if route.name in route_names:
                other = route_names[route.name]
                raise ConfigError(
                    f"route name {route.name!r} is used by both {other.file!r} "
                    f"(team {other.team!r}) and {filename!r} (team {team_file.team!r}); "
                    "route names must be unique across every team",
                    file=filename,
                )
            source = Source(file=filename, team=team_file.team)
            route_names[route.name] = source
            block = _compile_route(route, registry, source)
            route_blocks.append(f"# {route.name}  (team: {team_file.team}, source: {filename})\n{block}")

    toml = "\n\n".join([*header, *client_blocks, registry.render(), *route_blocks]).rstrip() + "\n"
    return CompileResult(toml=toml, route_names=route_names, api_key_envs=api_key_envs)


def compile_directory(teams_dir: Path, clients_path: Path | None = None) -> CompileResult:
    """Compile every ``*.yaml`` in ``teams_dir`` plus a clients file (default: ``teams_dir/../clients.yaml``)."""
    clients_path = clients_path or (teams_dir.parent / "clients.yaml")
    if not clients_path.exists():
        raise ConfigError(f"no clients.yaml found at {clients_path}", file=str(clients_path))
    clients = load_clients_file(clients_path)

    team_paths = sorted(p for p in teams_dir.glob("*.yaml"))
    if not team_paths:
        raise ConfigError(f"no team YAML files found in {teams_dir}", file=str(teams_dir))

    teams: list[tuple[str, TeamFile]] = []
    for path in team_paths:
        teams.append((str(path), load_team_file(path)))

    return compile_routes(clients, teams)


__all__ = [
    "CompileResult",
    "ConfigError",
    "Source",
    "TargetRegistry",
    "compile_directory",
    "compile_routes",
    "load_clients_file",
    "load_team_file",
]
