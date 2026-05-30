"""Use the LLM to produce a generator specification from literature priors.
Thorough variant of pretest_d: multiple temperature samples to inspect LLM variance."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from config import GROQ_API_KEY_ENV, OCM_METABOLITES, RESULTS_LLM_SPEC
from multi_agent.llm_clients import GroqClient
from multi_agent.prompts import LLM_SPEC_SYSTEM, LLM_SPEC_USER_TEMPLATE
from synth.llm_spec_loader import spec_summary

log = logging.getLogger(__name__)

OUT_DIR = RESULTS_LLM_SPEC
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_spec(metabolites: tuple = OCM_METABOLITES,
                  n_samples: int = 3,
                  temperature: float = 0.2,
                  out_path: Path | None = None
                  ) -> list[dict]:
    """Generate `n_samples` LLM specs. Each is saved to OUT_DIR/spec_<i>.json;
    the first valid one is also written as OUT_DIR/spec_canonical.json (used by run_comparison)."""
    if not os.environ.get(GROQ_API_KEY_ENV):
        raise RuntimeError(f"Set ${GROQ_API_KEY_ENV} to query the LLM")

    client = GroqClient()
    user_msg = LLM_SPEC_USER_TEMPLATE.format(
        metabolite_list="\n".join(f"  - {m}" for m in metabolites)
    )
    messages_template = [
        {"role": "system", "content": LLM_SPEC_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    specs: list[dict] = []
    canonical_set = False
    for i in range(n_samples):
        log.info("LLM spec sample %d/%d (T=%g)", i + 1, n_samples, temperature)
        try:
            raw = client.chat(messages_template,
                               temperature=temperature, max_tokens=2000)
        except Exception as e:
            log.error("Sample %d failed: %s", i, e)
            specs.append({})
            continue

        raw_path = OUT_DIR / f"spec_{i:02d}_raw.txt"
        raw_path.write_text(raw, encoding="utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}
        specs.append(parsed)

        if parsed:
            (OUT_DIR / f"spec_{i:02d}.json").write_text(
                json.dumps(parsed, indent=2), encoding="utf-8")
            log.info("  summary: %s", spec_summary(parsed))
            if not canonical_set:
                (OUT_DIR / "spec_canonical.json").write_text(
                    json.dumps(parsed, indent=2), encoding="utf-8")
                canonical_set = True

    if out_path:
        with out_path.open("w", encoding="utf-8") as f:
            for i, spec in enumerate(specs):
                f.write(json.dumps({
                    "i": i,
                    "summary": spec_summary(spec),
                }) + "\n")

    return specs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    specs = generate_spec(n_samples=args.n_samples, temperature=args.temperature)
    print(f"Got {len(specs)} specs; first non-empty saved to {OUT_DIR}/spec_canonical.json")
