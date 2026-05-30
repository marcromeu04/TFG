"""Benchmark v1 (single-LLM) vs v2 (heterogeneous) multi-agent runs.
Computes final gap closure, schema validity, hallucination, reasoning consistency, self-correction."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from multi_agent.orchestrator import OrchestratorRun

log = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    final_gap_closure: float
    n_iterations: int
    schema_validity_rate: float
    hallucination_rate: float
    reasoning_consistency: float
    self_correction_rate: float


def compute_run_metrics(run: OrchestratorRun,
                         metric_name: str = "spearman_mean"
                         ) -> AgentMetrics:
    """Compute the six agent-quality metrics for a single run."""
    n_it = len(run.iterations)
    if n_it == 0:
        return AgentMetrics(0.0, 0, 0.0, 0.0, 0.0, 0.0)

    initial = float(run.initial_metrics.get(metric_name, 0.0))
    final = float(run.final_metrics.get(metric_name, initial)
                  if run.final_metrics else initial)
    final_gap_closure = final - initial

    n_steps = 0
    n_valid = 0
    for r in run.iterations:
        for slot in ("diagnostician", "strategist", "executor"):
            n_steps += 1
            if r.__getattribute__(slot).get("error") is None and \
               r.__getattribute__(slot).get("parsed"):
                n_valid += 1
        if r.critic is not None:
            n_steps += 1
            if r.critic.get("error") is None and r.critic.get("parsed"):
                n_valid += 1
    schema_validity = n_valid / max(n_steps, 1)

    n_diag = 0
    n_halluc = 0
    for r in run.iterations:
        diag = r.diagnostician.get("parsed", {})
        if "evidence" not in diag:
            continue
        n_diag += 1
        evidences = diag.get("evidence", [])
        keys_available = set(r.metrics_before.keys())
        cited_any = False
        for ev in evidences:
            if isinstance(ev, str):
                for k in keys_available:
                    if k in ev:
                        cited_any = True
                        break
                if cited_any:
                    break
        if not cited_any:
            n_halluc += 1
    hallucination = n_halluc / max(n_diag, 1)

    n_pairs = 0
    n_inconsistent = 0
    for i in range(1, len(run.iterations)):
        prev = run.iterations[i - 1].diagnostician.get("parsed", {})
        cur = run.iterations[i].diagnostician.get("parsed", {})
        if not prev or not cur:
            continue
        n_pairs += 1
        prev_metrics = run.iterations[i - 1].metrics_after or run.iterations[i - 1].metrics_before
        cur_metrics = run.iterations[i].metrics_before
        delta = abs(float(cur_metrics.get(metric_name, 0)) -
                    float(prev_metrics.get(metric_name, 0)))
        if delta < 0.02 and prev.get("diagnosis") != cur.get("diagnosis"):
            n_inconsistent += 1
    consistency = 1.0 - (n_inconsistent / max(n_pairs, 1))

    n_critics = 0
    n_overrides = 0
    for r in run.iterations:
        if r.critic is None:
            continue
        n_critics += 1
        decision = r.critic.get("parsed", {}).get("decision", "")
        if decision in ("pivot", "stop"):
            n_overrides += 1
    self_corr = n_overrides / max(n_critics, 1)

    return AgentMetrics(
        final_gap_closure=final_gap_closure,
        n_iterations=n_it,
        schema_validity_rate=schema_validity,
        hallucination_rate=hallucination,
        reasoning_consistency=consistency,
        self_correction_rate=self_corr,
    )


def benchmark_runs(runs_v1: list[OrchestratorRun],
                    runs_v2: list[OrchestratorRun],
                    metric_name: str = "spearman_mean"
                    ) -> dict:
    """Compare two collections of runs (v1 vs v2). Returns per-version means + Wilcoxon."""
    m_v1 = [compute_run_metrics(r, metric_name) for r in runs_v1]
    m_v2 = [compute_run_metrics(r, metric_name) for r in runs_v2]

    df_v1 = pd.DataFrame([m.__dict__ for m in m_v1])
    df_v2 = pd.DataFrame([m.__dict__ for m in m_v2])

    out = {"v1": df_v1.mean().to_dict(), "v2": df_v2.mean().to_dict()}

    if len(df_v1) >= 5 and len(df_v2) >= 5:
        # Paired Wilcoxon on final_gap_closure (assumes paired runs).
        try:
            n = min(len(df_v1), len(df_v2))
            stat, p = wilcoxon(df_v1["final_gap_closure"].values[:n],
                               df_v2["final_gap_closure"].values[:n])
            out["wilcoxon_final_gap_closure"] = {"stat": float(stat),
                                                  "p": float(p)}
        except Exception as e:
            log.warning("wilcoxon failed: %s", e)
    return out


def load_run_from_jsonl(path: Path) -> OrchestratorRun:
    """Reconstruct an OrchestratorRun from a JSONL audit log."""
    from multi_agent.orchestrator import IterationRecord
    iters = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            iters.append(IterationRecord(**d))
    initial_metrics = iters[0].metrics_before if iters else {}
    final_metrics = (iters[-1].metrics_after if iters and iters[-1].metrics_after
                     else initial_metrics)
    return OrchestratorRun(
        prompts_version="?",
        initial_metrics=initial_metrics,
        iterations=iters,
        final_metrics=final_metrics,
    )
