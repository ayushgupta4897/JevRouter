"""The YAML schema teams write, and nothing more than that.

Two kinds of file:

* ``clients.yaml`` (one, platform-managed): the shared upstream connections routes draw on.
* ``teams/<team>.yaml`` (one per team, self-serve): a team's own routes.

A route picks one of four policies -- the same four shapes Switchyard's judge-backed algorithms
already support, given friendlier names and defaults:

* ``auto``       -- zero-config default (Switchyard's own `auto` preset).
* ``complexity`` -- judge forecasts whether the weak model can do it; threshold decides.
* ``escalation`` -- start weak; a judge watching the transcript escalates to strong after
                    repeated trouble.
* ``intent``     -- judge picks one of N named model buckets by content (the voice-AI
                    "extraction vs. analysis" case). At least two buckets.

Every judge-backed route gets Jev (this project's shared ``jevjudge`` sidecar) as its classifier
by default -- teams don't need to know Jev exists to get a cheap, fast judge. A team can name
its own judge client/model instead via ``judge:``.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROUTE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-/]*$")
BUCKET_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
JEV_JUDGE_CLIENT = "jevjudge"
JEV_JUDGE_MODEL = "jev-latest"


class ConfigError(ValueError):
    """A team or platform YAML file failed validation. `.file` is set by the caller if known."""

    def __init__(self, message: str, file: str | None = None):
        super().__init__(message)
        self.file = file


class ClientDef(BaseModel):
    """One upstream connection, shared across every team that names it. Platform-managed."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["openai_chat", "openai_responses", "anthropic_messages"] = "openai_chat"
    base_url: str
    api_key_env: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = Field(default=None, ge=0, le=10)


class ModelPricing(BaseModel):
    """What one model costs, in USD per 1M tokens -- only needed for cache-aware routing, which
    has to price the cache a model switch would throw away. Platform-managed, like clients."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    cached_input: float = Field(ge=0, description="Price of input tokens served from the prompt cache.")
    output: float = Field(ge=0)
    cache_write: float | None = Field(
        default=None, ge=0,
        description="Price of input tokens written into the cache, for providers that charge a "
        "premium for it (Anthropic: 1.25x input for the 5-minute cache; OpenAI GPT-5.6+: 1.25x). "
        "Omit where writes cost the normal input price (older OpenAI models, Gemini implicit).",
    )
    cache_ttl_seconds: int = Field(default=300, ge=1, description="Idle time after which the cache is assumed cold.")
    min_cacheable_tokens: int = Field(default=1024, ge=0, description="Prefixes shorter than this are never cached.")

    @model_validator(mode="after")
    def _cached_not_above_input(self) -> "ModelPricing":
        if self.cached_input > self.input:
            raise ConfigError(f"cached_input ({self.cached_input}) cannot exceed input ({self.input})")
        return self


class ClientsFile(BaseModel):
    """The platform-managed ``clients.yaml``: every upstream a team's routes may reference."""

    model_config = ConfigDict(extra="forbid")

    clients: dict[str, ClientDef]
    pricing: dict[str, ModelPricing] = Field(default_factory=dict, description="Keyed by model id.")

    @model_validator(mode="after")
    def _reserved_name(self) -> "ClientsFile":
        if JEV_JUDGE_CLIENT in self.clients:
            raise ConfigError(
                f"clients.{JEV_JUDGE_CLIENT!r} is reserved: routerctl injects the shared Jev "
                "judge client automatically. Pick a different name if you meant something else."
            )
        return self


class ModelRef(BaseModel):
    """One named model a route can send traffic to."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Exact model ID sent upstream.")
    client: str = Field(min_length=1, description="A key from clients.yaml.")
    description: str | None = Field(
        default=None,
        description="Required for `intent` buckets: what routes here, in one or two sentences. "
        "Becomes the judge's own criterion for this bucket, so specificity here is what makes "
        "intent routing accurate -- vague descriptions get vague routing.",
    )
    extra_body: dict[str, object] | None = None


class JudgeOverride(BaseModel):
    """Override the default shared Jev judge for one route."""

    model_config = ConfigDict(extra="forbid")

    id: str = JEV_JUDGE_MODEL
    client: str = JEV_JUDGE_CLIENT


class AutoPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["auto"]
    models: dict[Literal["efficient", "capable"], ModelRef]

    @model_validator(mode="after")
    def _both_tiers(self) -> "AutoPolicy":
        missing = {"efficient", "capable"} - self.models.keys()
        if missing:
            raise ConfigError(f"policy: auto needs models.{{{', '.join(sorted(missing))}}}")
        return self


class ComplexityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["complexity"]
    models: dict[Literal["weak", "strong"], ModelRef]
    base_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold_step: float = Field(default=0.1, ge=0.0)
    judge: JudgeOverride = Field(default_factory=JudgeOverride)

    @model_validator(mode="after")
    def _both_tiers(self) -> "ComplexityPolicy":
        missing = {"weak", "strong"} - self.models.keys()
        if missing:
            raise ConfigError(f"policy: complexity needs models.{{{', '.join(sorted(missing))}}}")
        if self.base_threshold + 2 * self.threshold_step > 1.0 + 1e-9:
            raise ConfigError(
                f"base_threshold ({self.base_threshold}) + 2 * threshold_step "
                f"({self.threshold_step}) must not exceed 1.0"
            )
        return self


class EscalationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["escalation"]
    models: dict[Literal["weak", "strong"], ModelRef]
    confirmations: int = Field(default=2, ge=1)
    recent_turn_window: int = Field(default=28, ge=1)
    judge: JudgeOverride = Field(default_factory=JudgeOverride)

    @model_validator(mode="after")
    def _both_tiers(self) -> "EscalationPolicy":
        missing = {"weak", "strong"} - self.models.keys()
        if missing:
            raise ConfigError(f"policy: escalation needs models.{{{', '.join(sorted(missing))}}}")
        return self


class IntentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["intent"]
    models: dict[str, ModelRef]
    default: str = Field(description="Bucket used when the judge is unsure or unreachable.")
    judge: JudgeOverride = Field(default_factory=JudgeOverride)

    @model_validator(mode="after")
    def _buckets(self) -> "IntentPolicy":
        if len(self.models) < 2:
            raise ConfigError(
                f"policy: intent needs at least 2 named buckets in models, got {len(self.models)}"
            )
        for name in self.models:
            if not BUCKET_NAME_RE.match(name):
                raise ConfigError(
                    f"bucket name {name!r} must start with a letter and contain only "
                    "letters, digits, '_' or '-' (it becomes a Jev choice label)"
                )
        missing_desc = [name for name, model in self.models.items() if not (model.description or "").strip()]
        if missing_desc:
            raise ConfigError(
                f"policy: intent buckets need a `description` the judge can route on; "
                f"missing for: {', '.join(sorted(missing_desc))}"
            )
        if self.default not in self.models:
            raise ConfigError(f"default: {self.default!r} is not one of models.{{{', '.join(self.models)}}}")
        return self


RoutePolicy = Annotated[
    AutoPolicy | ComplexityPolicy | EscalationPolicy | IntentPolicy,
    Field(discriminator="policy"),
]


class CacheConfig(BaseModel):
    """Cache-aware switching for one route (decision-only mode). Inert unless a request says
    which model served the session's previous turn -- see `routerctl/cache.py`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    horizon_turns: int = Field(
        default=5, ge=1,
        description="How many upcoming turns a switch's cost or savings is weighed over.",
    )
    expected_output_tokens: int = Field(default=500, ge=0, description="Per-turn output estimate, for pricing a turn.")
    switch_margin: float = Field(
        default=0.10, ge=0, le=1,
        description="A downgrade must save at least this fraction of staying's cost over the horizon.",
    )
    upgrade_min_confidence: float = Field(
        default=0.2, ge=0, le=1,
        description="An upgrade that would drop a warm cache needs a judge verdict at least this "
        "decisive; a borderline verdict holds the current model instead.",
    )


_ROUTE_LEVEL_KEYS = {"name", "policy", "cache"}


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The model ID clients send, e.g. 'deployment/transcription'.")
    policy: RoutePolicy
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: object) -> object:
        # `policy: intent` in YAML is both the discriminator field on the route *and* the tag
        # pydantic's discriminated union reads -- so the route's other fields (models, default,
        # judge, ...) live at the same level as `policy`, not nested under a `spec:` key. This
        # reshapes {name, policy, **rest} -> {name, policy: {policy: <tag>, **rest}} once, so
        # both the human-facing YAML and the discriminated union stay simple. Route-level keys
        # (`cache`) stay on the route, not the policy.
        if not isinstance(data, dict) or "policy" not in data:
            return data
        if isinstance(data["policy"], dict):
            return data
        policy_tag = data["policy"]
        rest = {k: v for k, v in data.items() if k not in _ROUTE_LEVEL_KEYS}
        reshaped = {"name": data.get("name"), "policy": {"policy": policy_tag, **rest}}
        if "cache" in data:
            reshaped["cache"] = data["cache"]
        return reshaped

    @model_validator(mode="after")
    def _valid_name(self) -> "Route":
        if not ROUTE_NAME_RE.match(self.name):
            raise ConfigError(
                f"route name {self.name!r} must start with a letter or digit and contain only "
                "letters, digits, '_', '-', '.', or '/'"
            )
        return self


class TeamFile(BaseModel):
    """One team's ``teams/<team>.yaml``: everything they own and can change themselves."""

    model_config = ConfigDict(extra="forbid")

    team: str = Field(min_length=1)
    owner: str | None = None
    routes: list[Route] = Field(min_length=1)
