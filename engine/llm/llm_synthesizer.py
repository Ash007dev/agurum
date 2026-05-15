"""
engine/llm/llm_synthesizer.py — LLM narrative synthesis for deep mode.

Uses claude-haiku-4-5 via the Anthropic SDK to generate a ≤300 word
SRE-style narrative explaining the reconstructed context.

Only invoked in deep mode. Fast mode uses the template narrative.
"""
from __future__ import annotations

import logging
from typing import Any

from engine import config

logger = logging.getLogger(__name__)


class LLMSynthesizer:
    """
    Async LLM-based narrative generator for deep mode reconstruction.

    Falls back to template narrative on any failure (API key missing,
    rate limit, timeout, etc.) — never causes a 500 error.
    """

    def __init__(self) -> None:
        self._client = None
        if config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=config.ANTHROPIC_API_KEY
                )
            except ImportError:
                logger.warning("anthropic SDK not installed — deep mode will use template narrative")
        else:
            logger.info("No ANTHROPIC_API_KEY — deep mode will use template narrative")

    @property
    def available(self) -> bool:
        return self._client is not None

    async def synthesize(
        self,
        signal: dict,
        window_events: list[dict],
        causal_edges: list[dict],
        similar_incidents: list[dict],
        remediations: list[dict],
    ) -> str:
        """
        Generate a deep-mode SRE narrative using claude-haiku-4-5.

        Returns template narrative on failure.
        """
        if not self._client:
            return self._template_fallback(signal, window_events, similar_incidents, remediations)

        prompt = self._build_prompt(
            signal, window_events, causal_edges, similar_incidents, remediations
        )

        try:
            response = await self._client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            return text.strip() or self._template_fallback(
                signal, window_events, similar_incidents, remediations
            )
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return self._template_fallback(signal, window_events, similar_incidents, remediations)

    def _build_prompt(
        self,
        signal: dict,
        window_events: list[dict],
        causal_edges: list[dict],
        similar_incidents: list[dict],
        remediations: list[dict],
    ) -> str:
        """Build the SRE narrative prompt."""
        svc = signal.get("service", "unknown")
        trigger = signal.get("trigger", "unknown")
        n_events = len(window_events)
        n_similar = len(similar_incidents)

        causal_summary = ""
        for edge in causal_edges[:5]:
            causal_summary += f"  - {edge.get('evidence', '?')} (conf={edge.get('confidence', 0):.2f})\n"

        similar_summary = ""
        for inc in similar_incidents[:3]:
            similar_summary += (
                f"  - {inc.get('incident_id', '?')} "
                f"(similarity={inc.get('similarity', 0):.3f})\n"
            )

        remediation_summary = ""
        for rem in remediations[:3]:
            remediation_summary += (
                f"  - {rem.get('action', '?')} on {rem.get('target', '?')} "
                f"(conf={rem.get('confidence', 0):.2f}, "
                f"history={rem.get('historical_outcome', '?')})\n"
            )

        return f"""You are an SRE incident analyst. Write a concise ≤300 word narrative
explaining this incident context. Be direct and actionable.

INCIDENT:
- Service: {svc}
- Trigger: {trigger}
- Related events in 300s window: {n_events}

CAUSAL CHAIN:
{causal_summary or '  (no causal edges detected)'}

SIMILAR PAST INCIDENTS ({n_similar} matches):
{similar_summary or '  (no historical matches)'}

RECOMMENDED REMEDIATIONS:
{remediation_summary or '  (no recommendations)'}

Write the narrative now. Focus on: what happened, why it happened (causal chain),
what worked before (historical matches), and what to do now (remediation)."""

    def _template_fallback(
        self,
        signal: dict,
        window_events: list[dict],
        similar_incidents: list[dict],
        remediations: list[dict],
    ) -> str:
        """Fast template narrative — no LLM call."""
        svc = signal.get("service", "unknown")
        n_matches = len(similar_incidents)
        top_match = similar_incidents[0].get("incident_id", "none") if similar_incidents else "none"
        top_sim = similar_incidents[0].get("similarity", 0.0) if similar_incidents else 0.0
        action = remediations[0].get("action", "unknown") if remediations else "unknown"
        conf = remediations[0].get("confidence", 0.0) if remediations else 0.0

        return (
            f"Incident on {svc} with {len(window_events)} related events in 300s window. "
            f"Found {n_matches} similar past incidents; "
            f"best match: {top_match} (similarity={top_sim:.3f}). "
            f"Recommended action: {action} (confidence={conf:.2f})."
        )
