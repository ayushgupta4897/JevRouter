"""The judge: Jev first, with a confidence gate that cascades or abstains.

Given an OpenAI chat-completions request that asks for a JSON verdict:

1. extract the verdict schema (``response_format`` or the prompt-embedded schema)
2. compile it to Jev questions; build Jev state from the messages
3. ask Jev once (all questions in parallel, one round trip)
4. decode the answers into a schema-valid verdict and a gate confidence
5. if confidence is below the gate:
     * forward the *original* request to a fallback OpenAI-compatible judge, or
     * return a non-JSON reply so the router fails open (its configured default target), or
     * return Jev's verdict anyway, flagged (default: min_confidence = 0, so this never fires)
6. wrap the verdict as a chat completion; attach evidence under ``jevjudge``
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from .client import JevClient, JevClientConfig, JevError
from .compiler import CompileError, Compiled, compile_schema, decode_answers, extract_schema
from .state import build_state


@dataclass
class JudgeConfig:
    min_confidence: float = 0.0
    low_confidence_action: str = "return"  # return | fallback | abstain
    fallback_base_url: str = ""
    fallback_api_key: str = ""
    fallback_model: str = ""
    fallback_timeout_s: float = 60.0
    rubric_placement: str = "state"
    max_state_chars: int = 100_000
    bool_threshold: float = 0.5
    profile: str | None = None  # force a profile name; None = match on schema name
    abstain_text: str = "abstain: jevjudge confidence below gate"

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            min_confidence=float(os.environ.get("JEVJUDGE_MIN_CONFIDENCE", "0")),
            low_confidence_action=os.environ.get("JEVJUDGE_LOW_CONFIDENCE_ACTION", "return").strip().lower(),
            fallback_base_url=os.environ.get("JEVJUDGE_FALLBACK_BASE_URL", "").rstrip("/"),
            fallback_api_key=os.environ.get("JEVJUDGE_FALLBACK_API_KEY", ""),
            fallback_model=os.environ.get("JEVJUDGE_FALLBACK_MODEL", ""),
            fallback_timeout_s=float(os.environ.get("JEVJUDGE_FALLBACK_TIMEOUT_S", "60")),
            rubric_placement=os.environ.get("JEVJUDGE_RUBRIC_PLACEMENT", "state").strip().lower(),
            max_state_chars=int(os.environ.get("JEVJUDGE_MAX_STATE_CHARS", "100000")),
            bool_threshold=float(os.environ.get("JEVJUDGE_BOOL_THRESHOLD", "0.5")),
            profile=os.environ.get("JEVJUDGE_PROFILE") or None,
        )


@dataclass
class Stats:
    requests: int = 0
    jev_calls: int = 0
    jev_errors: int = 0
    fallbacks: int = 0
    abstains: int = 0
    low_confidence: int = 0
    jev_latency_ms_total: float = 0.0
    jev_input_tokens: int = 0
    by_schema: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "jev_calls": self.jev_calls,
            "jev_errors": self.jev_errors,
            "fallbacks": self.fallbacks,
            "abstains": self.abstains,
            "low_confidence": self.low_confidence,
            "jev_latency_ms_avg": (self.jev_latency_ms_total / self.jev_calls) if self.jev_calls else None,
            "jev_input_tokens": self.jev_input_tokens,
            "by_schema": dict(self.by_schema),
        }


@dataclass
class JudgeOutcome:
    response: dict[str, Any]
    source: str  # jev | fallback | abstain
    confidence: float | None
    latency_ms: float
    schema_name: str


class Judge:
    def __init__(self, jev: JevClient, config: JudgeConfig, http: httpx.AsyncClient | None = None) -> None:
        self.jev = jev
        self.config = config
        self.stats = Stats()
        self._http = http or httpx.AsyncClient(timeout=config.fallback_timeout_s)
        self._compiled_cache: dict[str, Compiled] = {}

    async def aclose(self) -> None:
        await self.jev.aclose()
        await self._http.aclose()

    # ------------------------------------------------------------------ public

    async def judge(self, request: dict[str, Any]) -> JudgeOutcome:
        self.stats.requests += 1
        started = time.perf_counter()
        messages = request.get("messages") or []
        state, system_prompt = build_state(
            messages,
            rubric_placement=self.config.rubric_placement,
            max_state_chars=self.config.max_state_chars,
        )
        schema, schema_name = extract_schema(request.get("response_format"), system_prompt)
        self.stats.by_schema[schema_name] = self.stats.by_schema.get(schema_name, 0) + 1

        compiled = self._compile(schema, schema_name, system_prompt)
        try:
            result = await self.jev.decide(state, compiled.questions)
        except JevError:
            self.stats.jev_errors += 1
            if self._fallback_configured():
                return await self._fallback(request, started, schema_name, reason="jev_error")
            raise
        self.stats.jev_calls += 1
        self.stats.jev_latency_ms_total += result.latency_ms
        self.stats.jev_input_tokens += int((result.usage or {}).get("input_tokens") or 0)

        decoded = decode_answers(compiled, result.answers, bool_threshold=self.config.bool_threshold)
        low = decoded.confidence < self.config.min_confidence
        if low:
            self.stats.low_confidence += 1
            action = self.config.low_confidence_action
            if action == "fallback" and self._fallback_configured():
                return await self._fallback(request, started, schema_name, reason="low_confidence", jev_confidence=decoded.confidence)
            if action == "abstain":
                self.stats.abstains += 1
                response = self._chat_response(
                    request, content=self.config.abstain_text, usage=result.usage,
                    meta={"source": "abstain", "confidence": decoded.confidence, "evidence": decoded.evidence, "jev_latency_ms": result.latency_ms},
                )
                return JudgeOutcome(response, "abstain", decoded.confidence, _ms(started), schema_name)

        meta = {
            "source": "jev",
            "model": result.model,
            "profile": compiled.profile,
            "confidence": decoded.confidence,
            "low_confidence": low,
            "evidence": decoded.evidence,
            "questions": list(compiled.questions.keys()),
            "jev_latency_ms": round(result.latency_ms, 2),
        }
        response = self._chat_response(request, content=json.dumps(decoded.verdict), usage=result.usage, meta=meta)
        return JudgeOutcome(response, "jev", decoded.confidence, _ms(started), schema_name)

    # ----------------------------------------------------------------- helpers

    def _compile(self, schema: dict[str, Any], schema_name: str, system_prompt: str) -> Compiled:
        key = json.dumps([schema, schema_name, system_prompt, self.config.profile], sort_keys=True)
        cached = self._compiled_cache.get(key)
        if cached is None:
            cached = compile_schema(
                schema,
                schema_name=schema_name,
                system_prompt=system_prompt,
                profile_name=self.config.profile,
                bool_threshold=self.config.bool_threshold,
            )
            if len(self._compiled_cache) > 256:
                self._compiled_cache.clear()
            self._compiled_cache[key] = cached
        return cached

    def _fallback_configured(self) -> bool:
        return bool(self.config.fallback_base_url and self.config.fallback_model)

    async def _fallback(self, request: dict[str, Any], started: float, schema_name: str, *, reason: str, jev_confidence: float | None = None) -> JudgeOutcome:
        self.stats.fallbacks += 1
        body = dict(request)
        body["model"] = self.config.fallback_model
        body["stream"] = False
        headers = {"Content-Type": "application/json"}
        if self.config.fallback_api_key:
            headers["Authorization"] = f"Bearer {self.config.fallback_api_key}"
        response = await self._http.post(f"{self.config.fallback_base_url}/chat/completions", json=body, headers=headers)
        if response.status_code >= 400:
            raise JevError(f"fallback judge returned {response.status_code}: {response.text[:300]}")
        payload = response.json()
        payload["jevjudge"] = {"source": "fallback", "reason": reason, "jev_confidence": jev_confidence, "fallback_model": self.config.fallback_model}
        return JudgeOutcome(payload, "fallback", jev_confidence, _ms(started), schema_name)

    @staticmethod
    def _chat_response(request: dict[str, Any], *, content: str, usage: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        prompt_tokens = int((usage or {}).get("input_tokens") or 0)
        completion_tokens = int((usage or {}).get("output_tokens") or 0)
        return {
            "id": f"chatcmpl-jevjudge-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(request.get("model") or "jevjudge"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "jevjudge": meta,
        }


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def build_judge_from_env() -> Judge:
    return Judge(JevClient(JevClientConfig.from_env()), JudgeConfig.from_env())


__all__ = ["Judge", "JudgeConfig", "JudgeOutcome", "Stats", "build_judge_from_env", "CompileError", "JevError"]
