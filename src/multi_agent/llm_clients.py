"""Thin Groq and OpenRouter wrappers with privacy validation, retry/backoff, rate limiting,
and JSONL audit logging. Both expose BaseLLMClient.chat(messages, model, temperature, max_tokens)."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import (
    GROQ_API_KEY_ENV,
    GROQ_MODEL_LARGE,
    GROQ_MODEL_SMALL,
    LOGS,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_MODEL_REASONING,
)
from multi_agent.privacy_filter import PrivacyViolation, validate_prompt

log = logging.getLogger(__name__)


Message = dict[str, str]      # {"role": "system|user|assistant", "content": "..."}


@dataclass
class CallRecord:
    """One LLM call's audit record."""
    timestamp: float
    provider: str
    model: str
    n_messages: int
    prompt_chars: int
    completion_chars: int
    latency_s: float
    error: Optional[str] = None


class BaseLLMClient:
    """Abstract base; subclasses override `_call_provider`."""

    provider_name: str = "base"

    def __init__(self,
                 *,
                 default_model: str,
                 log_path: Optional[Path] = None,
                 min_seconds_between_calls: float = 0.0):
        self.default_model = default_model
        self.log_path = Path(log_path) if log_path else None
        self.min_seconds_between_calls = min_seconds_between_calls
        self._last_call_t = 0.0
        self.records: list[CallRecord] = []

    def _rate_limit(self):
        if self.min_seconds_between_calls <= 0:
            return
        now = time.time()
        wait = self.min_seconds_between_calls - (now - self._last_call_t)
        if wait > 0:
            time.sleep(wait)

    def _check_messages_privacy(self, messages: list[Message]):
        """Validate every user/system message against privacy filter."""
        for m in messages:
            content = m.get("content", "")
            validate_prompt(content)

    def chat(self,
             messages: list[Message],
             *,
             model: Optional[str] = None,
             temperature: float = 0.0,
             max_tokens: int = 2000,
             max_retries: int = 3,
             backoff: float = 2.0
             ) -> str:
        """Send messages, return assistant reply. Retries on transient errors (not on PrivacyViolation)."""
        self._check_messages_privacy(messages)
        m = model or self.default_model
        self._rate_limit()

        last_err = None
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                reply = self._call_provider(messages, m, temperature, max_tokens)
                self._log_call(t0, m, messages, reply)
                self._last_call_t = time.time()
                return reply
            except PrivacyViolation:
                raise         # never retry a privacy violation.
            except Exception as e:
                last_err = str(e)
                log.warning("LLM call failed (attempt %d/%d) on %s/%s: %s",
                            attempt + 1, max_retries, self.provider_name, m, e)
                self._log_call(t0, m, messages, "", error=last_err)
                time.sleep(backoff * (attempt + 1))
        raise RuntimeError(
            f"LLM call to {self.provider_name}/{m} failed after "
            f"{max_retries} attempts: {last_err}"
        )

    def _call_provider(self,
                        messages: list[Message],
                        model: str,
                        temperature: float,
                        max_tokens: int) -> str:
        raise NotImplementedError

    def _log_call(self, t0: float,
                   model: str,
                   messages: list[Message],
                   reply: str,
                   error: Optional[str] = None):
        rec = CallRecord(
            timestamp=t0,
            provider=self.provider_name,
            model=model,
            n_messages=len(messages),
            prompt_chars=sum(len(m.get("content", "")) for m in messages),
            completion_chars=len(reply),
            latency_s=time.time() - t0,
            error=error,
        )
        self.records.append(rec)
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            entry = {
                **rec.__dict__,
                "messages": messages,
                "reply": reply,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class GroqClient(BaseLLMClient):
    provider_name = "groq"

    def __init__(self,
                 *,
                 api_key: Optional[str] = None,
                 default_model: str = GROQ_MODEL_LARGE,
                 log_path: Optional[Path] = None,
                 min_seconds_between_calls: float = 0.5):
        super().__init__(default_model=default_model,
                         log_path=log_path or LOGS / "groq_calls.jsonl",
                         min_seconds_between_calls=min_seconds_between_calls)
        self._api_key = api_key or os.environ.get(GROQ_API_KEY_ENV)
        if not self._api_key:
            raise RuntimeError(
                f"Groq API key not provided.  Set ${GROQ_API_KEY_ENV} or "
                f"pass api_key=... to GroqClient.")
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError(
                "groq package not installed; pip install groq") from e
        self._client = Groq(api_key=self._api_key)

    def _call_provider(self, messages, model, temperature, max_tokens):
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


# OpenRouter via OpenAI-compatible SDK.
class OpenRouterClient(BaseLLMClient):
    provider_name = "openrouter"

    def __init__(self,
                 *,
                 api_key: Optional[str] = None,
                 default_model: str = OPENROUTER_MODEL_REASONING,
                 log_path: Optional[Path] = None,
                 min_seconds_between_calls: float = 1.0):
        super().__init__(default_model=default_model,
                         log_path=log_path or LOGS / "openrouter_calls.jsonl",
                         min_seconds_between_calls=min_seconds_between_calls)
        self._api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV)
        if not self._api_key:
            raise RuntimeError(
                f"OpenRouter API key not provided.  Set "
                f"${OPENROUTER_API_KEY_ENV}.")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package not installed; pip install openai") from e
        self._client = OpenAI(api_key=self._api_key,
                               base_url="https://openrouter.ai/api/v1")

    def _call_provider(self, messages, model, temperature, max_tokens):
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


def get_client(provider: str = "groq", **kwargs) -> BaseLLMClient:
    """Return a configured client by name."""
    p = provider.lower()
    if p == "groq":
        return GroqClient(**kwargs)
    if p == "openrouter":
        return OpenRouterClient(**kwargs)
    raise ValueError(f"Unknown provider: {provider}; "
                     f"supported: groq, openrouter")
