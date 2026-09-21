"""Async client for System One decision endpoints.

Transports:

* ``typesafe``   POST {base}/v1/systemone            (TypeSafe native; also every open-source
                 clone that serves the same wire format: openjev-sglang, kev, litjev, Decider)
* ``openrouter`` POST {base}/api/alpha/decisions      (OpenRouter's Decisions router, alpha)

Both take ``{"model": .., "state": .., "questions": {..}}`` and return ``{"answers": {..}, "usage": ..}``.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


class JevError(RuntimeError):
    """The decision endpoint failed."""


@dataclass
class JevResult:
    answers: dict[str, Any]
    usage: dict[str, Any]
    model: str
    latency_ms: float
    raw: dict[str, Any]


@dataclass
class JevClientConfig:
    transport: str = "typesafe"
    base_url: str = "https://api.typesafe.ai"
    api_key: str = ""
    model: str = "jev-latest"
    timeout_s: float = 10.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "JevClientConfig":
        transport = os.environ.get("JEVJUDGE_TRANSPORT", "typesafe").strip().lower()
        if transport == "openrouter":
            base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai")
            key = os.environ.get("OPENROUTER_API_KEY", "")
            model = os.environ.get("JEVJUDGE_MODEL", "typesafe/jev-latest")
        else:
            base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
            key = os.environ.get("TYPESAFE_API_KEY", "")
            model = os.environ.get("JEVJUDGE_MODEL", os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest"))
        return cls(
            transport=transport,
            base_url=base.rstrip("/"),
            api_key=key,
            model=model,
            timeout_s=float(os.environ.get("JEVJUDGE_TIMEOUT_S", "10")),
            max_retries=int(os.environ.get("JEVJUDGE_MAX_RETRIES", "2")),
        )

    @property
    def endpoint(self) -> str:
        if self.transport == "openrouter":
            return f"{self.base_url}/api/alpha/decisions"
        return f"{self.base_url}/v1/systemone"


class JevClient:
    def __init__(self, config: JevClientConfig, http: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._http = http or httpx.AsyncClient(timeout=config.timeout_s)
        self._owned = http is None

    async def aclose(self) -> None:
        if self._owned:
            await self._http.aclose()

    async def decide(self, state: Any, questions: dict[str, dict[str, Any]], *, model: str | None = None) -> JevResult:
        body = {"model": model or self.config.model, "state": state, "questions": questions}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.transport == "openrouter":
            headers["X-Title"] = "jevjudge"
        attempt = 0
        started = time.perf_counter()
        while True:
            try:
                response = await self._http.post(self.config.endpoint, json=body, headers=headers)
            except httpx.HTTPError as error:
                if attempt >= self.config.max_retries:
                    raise JevError(f"transport error calling {self.config.endpoint}: {error}") from error
                attempt += 1
                await asyncio.sleep(0.2 * (2 ** (attempt - 1)))
                continue
            if response.status_code in {429, 529, 502, 503} and attempt < self.config.max_retries:
                attempt += 1
                await asyncio.sleep(0.2 * (2 ** (attempt - 1)))
                continue
            if response.status_code >= 400:
                raise JevError(f"{self.config.endpoint} returned {response.status_code}: {response.text[:500]}")
            payload = response.json()
            answers = payload.get("answers")
            if not isinstance(answers, dict):
                raise JevError("decision response has no `answers` object")
            return JevResult(
                answers=answers,
                usage=payload.get("usage") or {},
                model=str(payload.get("model") or body["model"]),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                raw=payload,
            )
