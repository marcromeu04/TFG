"""Validate outgoing LLM prompts; raise PrivacyViolation if patient-level data leaks.
Blacklist (config.PRIVACY_BLACKLIST_PATTERNS) plus heuristic: a dense run of floats looks like raw spectra."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from config import PRIVACY_BLACKLIST_PATTERNS

log = logging.getLogger(__name__)


class PrivacyViolation(Exception):
    """Raised when a prompt fails the privacy filter."""


@dataclass
class PrivacyCheckResult:
    ok: bool
    violations: list[str]


# heurística: a stretch of >= MAX_FLOAT_DENSITY numbers within
# WINDOW_CHARS characters is flagged as "raw data"
MAX_FLOAT_DENSITY = 30
WINDOW_CHARS = 200
FLOAT_RE = re.compile(r"-?\d+\.\d+")


def _check_blacklist(text: str) -> list[str]:
    """Return matched blacklisted substrings (empty if none)."""
    found = []
    for pat in PRIVACY_BLACKLIST_PATTERNS:
        try:
            r = re.compile(pat, re.IGNORECASE)
            for m in r.finditer(text):
                found.append(m.group(0))
        except re.error:
            log.warning("Bad regex in PRIVACY_BLACKLIST_PATTERNS: %r", pat)
    return found


def _check_raw_data_density(text: str) -> bool:
    """True if a long run of floats looks like raw data."""
    matches = list(FLOAT_RE.finditer(text))
    if len(matches) < MAX_FLOAT_DENSITY:
        return False
    for i in range(len(matches) - MAX_FLOAT_DENSITY + 1):
        span = matches[i + MAX_FLOAT_DENSITY - 1].start() - matches[i].start()
        if span <= WINDOW_CHARS:
            return True
    return False


def check_prompt(text: str) -> PrivacyCheckResult:
    """Inspect a prompt and return a PrivacyCheckResult."""
    if not isinstance(text, str):
        return PrivacyCheckResult(ok=False,
                                  violations=["prompt is not a string"])
    violations: list[str] = []
    blocked = _check_blacklist(text)
    if blocked:
        violations.append(f"blacklisted patterns matched: {set(blocked)}")
    if _check_raw_data_density(text):
        violations.append(
            f"too many floats in narrow window "
            f"(>{MAX_FLOAT_DENSITY} in {WINDOW_CHARS} chars); "
            "looks like raw spectral data"
        )
    return PrivacyCheckResult(ok=not violations, violations=violations)


def validate_prompt(text: str, *, raise_on_fail: bool = True) -> bool:
    """Validate a prompt; raises PrivacyViolation (or returns False) on failure."""
    res = check_prompt(text)
    if not res.ok:
        if raise_on_fail:
            raise PrivacyViolation(
                "Prompt rejected by privacy filter:\n  - "
                + "\n  - ".join(res.violations)
            )
        log.warning("Privacy violation (not raised): %s", res.violations)
        return False
    return True
