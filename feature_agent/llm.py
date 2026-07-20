"""OpenRouter client wrapper — the only module that touches the network.

OpenRouter exposes an OpenAI-compatible API, so we drive it with the `openai`
SDK pointed at OpenRouter's base URL. The SDK is imported lazily so the entire
deterministic pipeline (profile, execute, evaluate, select-by-gates) runs with no
`openai` install and no API key — the LLM is judgment on top of deterministic
statistics, never a hard dependency.

Responsibilities (all three LLM call sites go through here):
  * model from config, never hardcoded; per-call temperature;
  * structured output: request JSON, validate into a Pydantic model, and retry
    up to `max_retries` with the validation error fed back into the prompt;
  * budget accounting: token + estimated-cost ceiling enforced per run;
  * a call log (prompt/response hashes, tokens, cost) recorded in the manifest so
    a run is auditable and — with a pinned model + fixtures — reproducible.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .config import FeatureAgentConfig

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The LLM layer cannot be used (missing dep, key, or an API failure)."""


class BudgetExceeded(RuntimeError):
    """The per-run cost ceiling would be exceeded by the next call."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class CallRecord:
    stage: str
    model: str
    prompt_sha: str
    response_sha: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    attempts: int


@dataclass
class Budget:
    """Running tally of LLM spend, checked against the config ceiling."""

    max_cost_usd: float
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def would_exceed(self) -> bool:
        return self.cost_usd >= self.max_cost_usd

    def add(self, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd
        self.calls += 1


class LLMClient:
    def __init__(self, config: FeatureAgentConfig):
        self.config = config
        self.budget = Budget(max_cost_usd=config.max_cost_usd)
        self.call_log: list[CallRecord] = []
        self._client = None  # created lazily

    # ------------------------------------------------------------------ #
    def available(self) -> tuple[bool, str]:
        """Can we make LLM calls? Returns (ok, reason-if-not)."""
        if not self.config.use_llm:
            return False, "use_llm is disabled in config."
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "the 'openai' package is not installed."
        if not self.config.openrouter_api_key:
            return False, "OPENROUTER_API_KEY is not set."
        return True, ""

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        ok, why = self.available()
        if not ok:
            raise LLMError(why)
        from openai import OpenAI

        headers: dict[str, str] = {}
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name

        self._client = OpenAI(
            base_url=self.config.openrouter_base_url,
            api_key=self.config.openrouter_api_key,
            timeout=self.config.request_timeout,
            default_headers=headers or None,
        )
        return self._client

    # ------------------------------------------------------------------ #
    def _estimate_cost(self, usage: Any) -> float:
        """Prefer the API-reported cost; otherwise estimate from token counts."""
        reported = getattr(usage, "cost", None)
        if reported is not None:
            try:
                return float(reported)
            except (TypeError, ValueError):
                pass
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        return (
            pt / 1_000_000 * self.config.input_price_per_mtok
            + ct / 1_000_000 * self.config.output_price_per_mtok
        )

    def _chat(self, messages: list[dict], temperature: float, model: str,
              max_retries: int = 4) -> tuple[str, Any]:
        """One completion with retry/backoff. Returns (content, usage)."""
        if self.budget.would_exceed():
            raise BudgetExceeded(
                f"Cost ceiling ${self.config.max_cost_usd:.2f} reached "
                f"(spent ${self.budget.cost_usd:.4f})."
            )
        client = self._ensure_client()
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.config.max_output_tokens,
            response_format={"type": "json_object"},
            seed=self.config.random_state,  # best-effort determinism
            extra_body={"usage": {"include": True}},  # ask OpenRouter for exact cost
        )
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                if not resp.choices:
                    raise LLMError(f"Model '{model}' returned no choices.")
                return resp.choices[0].message.content or "", getattr(resp, "usage", None)
            except LLMError:
                raise
            except Exception as exc:  # transient network / rate-limit / 5xx
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status is not None and status != 429 and 400 <= status < 500:
                    raise LLMError(f"OpenRouter request failed ({status}): {exc}") from exc
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"OpenRouter request failed after {max_retries} attempts: {last_exc}")

    # ------------------------------------------------------------------ #
    def structured(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float,
        model: str | None = None,
        max_retries: int = 2,
    ) -> T:
        """Get a schema-validated Pydantic object from the model.

        Requests JSON, validates against `schema`, and on a validation error
        retries (up to `max_retries` extra attempts) with the error appended so
        the model can correct itself. Hard-fails the *batch* (raises), not the run
        — the orchestrator catches this and degrades gracefully.
        """
        model = model or self.config.model
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        base_user = (
            f"{user}\n\n"
            f"Respond with a single JSON object that validates against this JSON Schema. "
            f"Output JSON only — no prose, no markdown fences.\n\nJSON Schema:\n{schema_json}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": base_user},
        ]
        last_err = ""
        for attempt in range(max_retries + 1):
            content, usage = self._chat(messages, temperature=temperature, model=model)
            # account for spend regardless of whether parsing succeeds
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            cost = self._estimate_cost(usage)
            self.budget.add(pt, ct, cost)
            self.call_log.append(CallRecord(
                stage=stage, model=model,
                prompt_sha=_sha(json.dumps(messages, sort_keys=True, default=str)),
                response_sha=_sha(content), prompt_tokens=pt,
                completion_tokens=ct, cost_usd=round(cost, 6), attempts=attempt + 1,
            ))
            try:
                obj = _parse_json(content)
                return schema.model_validate(obj)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_err = str(exc)
                if attempt < max_retries:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"That response failed validation:\n{last_err}\n\n"
                            f"Return corrected JSON only, matching the schema exactly."
                        ),
                    })
        raise LLMError(f"[{stage}] model output failed validation after "
                       f"{max_retries + 1} attempts: {last_err}")


def _parse_json(content: str) -> Any:
    """Parse JSON, tolerating stray markdown fences some models still emit."""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    s = s.strip()
    # if extra prose leaked in, grab the outermost JSON object
    if not s.startswith("{"):
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end + 1]
    return json.loads(s)
