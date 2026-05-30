"""Multi-agent diagnostic system: agents, prompts, orchestrator,
privacy filter, LLM clients, benchmark."""
from multi_agent.privacy_filter import (
    PrivacyViolation,
    check_prompt,
    validate_prompt,
)
from multi_agent.llm_clients import (
    BaseLLMClient,
    GroqClient,
    OpenRouterClient,
    get_client,
)
from multi_agent.prompts import (
    PROMPTS_V1,
    PROMPTS_V2,
    PromptVersion,
    LLM_SPEC_SYSTEM,
    LLM_SPEC_USER_TEMPLATE,
)
from multi_agent.agents import (
    Agent,
    AgentReply,
    Critic,
    Diagnostician,
    Executor,
    Strategist,
)
from multi_agent.orchestrator import (
    IterationRecord,
    OrchestratorRun,
    run_orchestrator,
)
from multi_agent.benchmark import (
    AgentMetrics,
    benchmark_runs,
    compute_run_metrics,
    load_run_from_jsonl,
)

__all__ = [
    "PrivacyViolation", "check_prompt", "validate_prompt",
    "BaseLLMClient", "GroqClient", "OpenRouterClient", "get_client",
    "PROMPTS_V1", "PROMPTS_V2", "PromptVersion",
    "LLM_SPEC_SYSTEM", "LLM_SPEC_USER_TEMPLATE",
    "Agent", "AgentReply", "Critic", "Diagnostician",
    "Executor", "Strategist",
    "IterationRecord", "OrchestratorRun", "run_orchestrator",
    "AgentMetrics", "benchmark_runs", "compute_run_metrics",
    "load_run_from_jsonl",
]
