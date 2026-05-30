"""Agent classes for the four-role diagnostic system (Diagnostician, Strategist, Executor, Critic).
Each agent's step() sends [system, user] to the LLM, parses the JSON reply, and validates against
an expected schema (missing keys raise; unknown keys are ignored)."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from multi_agent.llm_clients import BaseLLMClient

log = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM output: direct, fenced ```json```, or first balanced {...}."""
    if not isinstance(text, str):
        raise ValueError(f"Expected string, got {type(text)}")

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    break

    raise ValueError(f"Could not extract JSON from reply (first 200 "
                     f"chars): {text[:200]}")


@dataclass
class AgentReply:
    """Structured reply with raw text and parsed JSON."""
    raw: str
    parsed: dict
    error: Optional[str] = None


class Agent:
    """Base class for typed agents."""

    role: str = "agent"
    expected_keys: tuple = ()

    def __init__(self,
                 client: BaseLLMClient,
                 system_prompt: str,
                 model: str,
                 *,
                 temperature: float = 0.0,
                 max_tokens: int = 1500):
        self.client = client
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_user_message(self, **inputs) -> str:
        raise NotImplementedError

    def _validate(self, parsed: dict) -> Optional[str]:
        """Return None if OK, error message if a required key is missing."""
        for k in self.expected_keys:
            if k not in parsed:
                return f"missing key: {k}"
        return None

    def step(self, **inputs) -> AgentReply:
        user_msg = self._build_user_message(**inputs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            raw = self.client.chat(messages,
                                    model=self.model,
                                    temperature=self.temperature,
                                    max_tokens=self.max_tokens)
        except Exception as e:
            return AgentReply(raw="", parsed={}, error=str(e))
        try:
            parsed = _extract_json(raw)
        except ValueError as e:
            return AgentReply(raw=raw, parsed={}, error=str(e))
        err = self._validate(parsed)
        return AgentReply(raw=raw, parsed=parsed, error=err)


class Diagnostician(Agent):
    role = "diagnostician"
    expected_keys = ("diagnosis", "confidence", "reasoning")

    def _build_user_message(self, *,
                            current_metrics: dict,
                            current_config: dict,
                            cohort_size: int,
                            iteration: int):
        return (
            f"Iteration: {iteration}\n"
            f"Cohort size (real): {cohort_size}\n\n"
            f"Latest aggregated metrics (mean across {cohort_size} samples,\n"
            f"averaged over the OCM panel of metabolites):\n"
            f"{json.dumps(current_metrics, indent=2)}\n\n"
            f"Current generator config (selected fields):\n"
            f"{json.dumps(current_config, indent=2)}\n\n"
            f"Diagnose the dominant residual error mode."
        )


class Strategist(Agent):
    role = "strategist"
    expected_keys = ("actions", "rationale")

    def _build_user_message(self, *,
                            diagnosis: dict,
                            recent_history: list[dict]):
        return (
            f"Latest diagnosis:\n"
            f"{json.dumps(diagnosis, indent=2)}\n\n"
            f"Recent action history "
            f"(last {len(recent_history)} iterations):\n"
            f"{json.dumps(recent_history, indent=2)}\n\n"
            f"Propose 1-2 actions."
        )


class Executor(Agent):
    role = "executor"
    expected_keys = ()       # any subset of override keys is acceptable.

    def _build_user_message(self, *, actions: list[str]):
        return (
            f"Strategist actions:\n"
            f"{json.dumps(actions, indent=2)}\n\n"
            f"Output the GeneratorConfig overrides as JSON."
        )


class Critic(Agent):
    role = "critic"
    expected_keys = ("decision", "rationale")

    def _build_user_message(self, *,
                            initial_gap: float,
                            current_gap: float,
                            history: list[dict]):
        return (
            f"Initial synth-real gap (Spearman ρ): {initial_gap:.3f}\n"
            f"Current synth-real gap (Spearman ρ): {current_gap:.3f}\n\n"
            f"Trajectory (last {len(history)} iterations):\n"
            f"{json.dumps(history, indent=2)}\n\n"
            f"Decide: continue, pivot, or stop."
        )
