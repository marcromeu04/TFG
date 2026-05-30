"""Main loop for the Diagnostician -> Strategist -> Executor -> Critic cycle.
Per iteration: D -> S -> E, then every CRITIC_EVERY iterations the Critic may stop or pivot.
Full message trajectory is persisted to JSONL for auditability."""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import LOGS, RESULTS_MULTIAGENT
from multi_agent.agents import (
    AgentReply,
    Critic,
    Diagnostician,
    Executor,
    Strategist,
)
from multi_agent.llm_clients import BaseLLMClient
from multi_agent.prompts import PROMPTS_V1, PROMPTS_V2, PromptVersion

log = logging.getLogger(__name__)

CRITIC_EVERY = 5  # todo: make configurable per run


@dataclass
class IterationRecord:
    iteration: int
    timestamp: float
    metrics_before: dict
    diagnostician: dict
    strategist: dict
    executor: dict
    metrics_after: Optional[dict] = None
    critic: Optional[dict] = None
    errors: list[str] = field(default_factory=list)


@dataclass
class OrchestratorRun:
    prompts_version: str
    initial_metrics: dict
    iterations: list[IterationRecord] = field(default_factory=list)
    final_metrics: Optional[dict] = None
    stopped_reason: Optional[str] = None
    duration_s: float = 0.0


def _resolve_model(client_assignment: dict, role_key: str) -> str:
    """Look up the model identifier for a role key ("large" or "small")."""
    return client_assignment[role_key]


def run_orchestrator(client: BaseLLMClient,
                     evaluate_fn: Callable[[dict], dict],
                     initial_config: dict,
                     initial_metrics: dict,
                     prompts: PromptVersion = PROMPTS_V1,
                     client_assignment: Optional[dict] = None,
                     max_iterations: int = 20,
                     log_path: Optional[Path] = None) -> OrchestratorRun:
    """Run the orchestrator. evaluate_fn(overrides_dict) -> metrics_dict decouples
    the agent loop from the eval pipeline."""
    if client_assignment is None:
        client_assignment = {"large": client.default_model,
                              "small": client.default_model}
    log_path = log_path or (RESULTS_MULTIAGENT /
                              f"orchestrator_{prompts.name}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    diag = Diagnostician(client, prompts.diagnostician_system,
                          model=_resolve_model(client_assignment,
                                                prompts.diagnostician_model))
    strat = Strategist(client, prompts.strategist_system,
                        model=_resolve_model(client_assignment,
                                              prompts.strategist_model))
    exec_ = Executor(client, prompts.executor_system,
                       model=_resolve_model(client_assignment,
                                             prompts.executor_model))
    critic = Critic(client, prompts.critic_system,
                     model=_resolve_model(client_assignment,
                                           prompts.critic_model))

    t_start = time.time()
    run = OrchestratorRun(prompts_version=prompts.name,
                           initial_metrics=initial_metrics)
    current_metrics = dict(initial_metrics)
    current_config = dict(initial_config)
    initial_gap = float(current_metrics.get("synth_real_gap_spearman", 0.0))

    for it in range(max_iterations):
        rec = IterationRecord(
            iteration=it,
            timestamp=time.time(),
            metrics_before=dict(current_metrics),
            diagnostician={},
            strategist={},
            executor={},
        )
        d = diag.step(current_metrics=current_metrics,
                       current_config=current_config,
                       cohort_size=current_metrics.get("cohort_size", 253),
                       iteration=it)
        rec.diagnostician = {"raw": d.raw, "parsed": d.parsed, "error": d.error}
        if d.error:
            rec.errors.append(f"diagnostician: {d.error}")
            run.iterations.append(rec)
            continue

        recent = [{"iter": r.iteration,
                    "diagnosis": r.diagnostician.get("parsed", {}).get("diagnosis"),
                    "actions": r.strategist.get("parsed", {}).get("actions"),
                    "metrics_after": r.metrics_after}
                  for r in run.iterations[-3:]]
        s = strat.step(diagnosis=d.parsed, recent_history=recent)
        rec.strategist = {"raw": s.raw, "parsed": s.parsed, "error": s.error}
        if s.error:
            rec.errors.append(f"strategist: {s.error}")
            run.iterations.append(rec)
            continue

        e = exec_.step(actions=s.parsed.get("actions", []))
        rec.executor = {"raw": e.raw, "parsed": e.parsed, "error": e.error}
        if e.error:
            rec.errors.append(f"executor: {e.error}")
            run.iterations.append(rec)
            continue

        overrides = e.parsed
        try:
            new_metrics = evaluate_fn(overrides)
            current_metrics = new_metrics
            current_config.update(overrides)
            rec.metrics_after = new_metrics
        except Exception as ex:
            rec.errors.append(f"evaluate_fn: {ex}")
            log.warning("Evaluation failed at iteration %d: %s", it, ex)

        if (it + 1) % CRITIC_EVERY == 0:
            current_gap = float(
                current_metrics.get("synth_real_gap_spearman", initial_gap))
            history = [{"iter": r.iteration,
                         "diagnosis": r.diagnostician.get("parsed", {}).get("diagnosis"),
                         "actions": r.strategist.get("parsed", {}).get("actions"),
                         "metrics_after": r.metrics_after}
                       for r in run.iterations[-CRITIC_EVERY:]]
            c = critic.step(initial_gap=initial_gap,
                             current_gap=current_gap,
                             history=history)
            rec.critic = {"raw": c.raw, "parsed": c.parsed, "error": c.error}
            decision = c.parsed.get("decision", "continue")
            if decision == "stop":
                run.stopped_reason = "critic_stop"
                run.iterations.append(rec)
                _flush_record(log_path, rec)
                break

        run.iterations.append(rec)
        _flush_record(log_path, rec)

    run.final_metrics = current_metrics
    run.duration_s = time.time() - t_start
    return run


def _flush_record(log_path: Path, rec: IterationRecord):
    """Append one iteration record to the JSONL audit log."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False, default=str) + "\n")
