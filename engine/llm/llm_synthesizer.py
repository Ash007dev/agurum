"""
engine/llm/llm_synthesizer.py — LLM narrative synthesis for deep mode.

Priority order:
  1. Groq (llama-3.3-70b-versatile via OpenAI SDK) — free, ~500ms
  2. Anthropic (claude-haiku-4-5) — paid, ~1s
  3. Template fallback — instant, no network

Only invoked in deep mode. Fast mode uses the enriched template narrative.
"""
from __future__ import annotations

import logging
from typing import Any

from engine import config

logger = logging.getLogger(__name__)


class LLMSynthesizer:
    """
    LLM-based narrative generator for deep mode reconstruction.

    Falls back gracefully: Groq → Anthropic → template.
    Never causes a crash — all errors are caught and logged.
    """

    def __init__(self) -> None:
        self._groq_client = None
        self._anthropic_client = None
        self._provider = "template"

        # Priority 1: Groq via OpenAI SDK
        if config.GROQ_API_KEY:
            try:
                from openai import OpenAI
                self._groq_client = OpenAI(
                    api_key=config.GROQ_API_KEY,
                    base_url=config.GROQ_BASE_URL,
                )
                self._provider = "groq"
                logger.info("LLM provider: Groq (llama-3.3-70b-versatile)")
            except ImportError:
                logger.warning("openai SDK not installed — trying Anthropic")

        # Priority 2: Anthropic
        if self._groq_client is None and config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self._anthropic_client = anthropic.AsyncAnthropic(
                    api_key=config.ANTHROPIC_API_KEY
                )
                self._provider = "anthropic"
                logger.info("LLM provider: Anthropic (claude-haiku-4-5)")
            except ImportError:
                logger.warning("anthropic SDK not installed — using template narrative")

        if self._provider == "template":
            logger.info("No LLM API key found — deep mode will use template narrative")

    @property
    def available(self) -> bool:
        return self._groq_client is not None or self._anthropic_client is not None

    async def synthesize(
        self,
        signal: dict,
        window_events: list[dict],
        causal_edges: list[dict],
        similar_incidents: list[dict],
        remediations: list[dict],
    ) -> str:
        """
        Generate a deep-mode SRE narrative.
        Tries Groq first, then Anthropic, then template.
        """
        prompt = self._build_prompt(
            signal, window_events, causal_edges, similar_incidents, remediations
        )

        # Try Groq (synchronous — OpenAI SDK)
        if self._groq_client:
            try:
                response = self._groq_client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": "You are an expert SRE incident analyst. Write concise, actionable incident narratives."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning(f"Groq synthesis failed: {e}")

        # Try Anthropic (async)
        if self._anthropic_client:
            try:
                response = await self._anthropic_client.messages.create(
                    model=config.LLM_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if response.content else ""
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning(f"Anthropic synthesis failed: {e}")

        # Template fallback
        return self._template_fallback(signal, window_events, similar_incidents, remediations)

    def synthesize_sync(
        self,
        signal: dict,
        window_events: list[dict],
        causal_edges: list[dict],
        similar_incidents: list[dict],
        remediations: list[dict],
    ) -> str:
        """
        Synchronous version for the benchmark adapter (no event loop).
        Only uses Groq (sync) or template. Anthropic is async-only.
        """
        prompt = self._build_prompt(
            signal, window_events, causal_edges, similar_incidents, remediations
        )

        if self._groq_client:
            try:
                response = self._groq_client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": "You are an expert SRE incident analyst. Write concise, actionable incident narratives."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
                if text.strip():
                    return text.strip()
            except Exception as e:
                logger.warning(f"Groq sync synthesis failed: {e}")

        return self._template_fallback(signal, window_events, similar_incidents, remediations)

    def _build_prompt(
        self,
        signal: dict,
        window_events: list[dict],
        causal_edges: list[dict],
        similar_incidents: list[dict],
        remediations: list[dict],
    ) -> str:
        """Build the SRE narrative prompt with full context."""
        svc = signal.get("service", "unknown")
        trigger = signal.get("trigger", "unknown")
        inc_id = signal.get("incident_id", "unknown")
        n_events = len(window_events)
        n_similar = len(similar_incidents)

        # Summarize event types
        kind_counts: dict[str, int] = {}
        for e in window_events:
            k = e.get("kind", "unknown")
            kind_counts[k] = kind_counts.get(k, 0) + 1
        event_breakdown = ", ".join(f"{v} {k}" for k, v in sorted(kind_counts.items()))

        # Causal chain summary
        causal_summary = ""
        for i, edge in enumerate(causal_edges[:8], 1):
            causal_summary += f"  {i}. {edge.get('evidence', '?')} (confidence={edge.get('confidence', 0):.2f})\n"

        # Similar incidents
        similar_summary = ""
        for inc in similar_incidents[:5]:
            similar_summary += (
                f"  - {inc.get('incident_id', '?')} "
                f"(similarity={inc.get('similarity', 0):.3f}, "
                f"rationale={inc.get('rationale', 'N/A')})\n"
            )

        # Remediation summary
        remediation_summary = ""
        for rem in remediations[:3]:
            remediation_summary += (
                f"  - {rem.get('action', '?')} on {rem.get('target', '?')} "
                f"(confidence={rem.get('confidence', 0):.2f}, "
                f"history={rem.get('historical_outcome', '?')})\n"
            )

        return f"""You are an SRE incident analyst writing a post-incident context report.
Write a structured ≤300 word narrative for this incident. Be direct, technical, and actionable.

INCIDENT: {inc_id}
- Service: {svc}
- Trigger: {trigger}
- Telemetry window: {n_events} events ({event_breakdown})

EXTRACTED CAUSAL CHAIN (chronological, source→effect):
{causal_summary or '  (no causal edges detected)'}

SIMILAR PAST INCIDENTS ({n_similar} matches):
{similar_summary or '  (no historical matches)'}

RECOMMENDED REMEDIATIONS:
{remediation_summary or '  (no recommendations)'}

Structure your narrative as:
1. **What happened** — timeline of the failure cascade
2. **Why it happened** — root cause from the causal chain
3. **Historical precedent** — what similar incidents tell us
4. **Recommended action** — specific remediation with confidence

Do NOT use markdown headers. Write flowing paragraphs. Be specific about service names, versions, and metrics."""

    def _template_fallback(
        self,
        signal: dict,
        window_events: list[dict],
        similar_incidents: list[dict],
        remediations: list[dict],
    ) -> str:
        """Rich template narrative — no LLM call."""
        svc = signal.get("service", "unknown")
        trigger = signal.get("trigger", "unknown")
        n_events = len(window_events)
        n_matches = len(similar_incidents)
        top_match = similar_incidents[0].get("incident_id", "none") if similar_incidents else "none"
        top_sim = similar_incidents[0].get("similarity", 0.0) if similar_incidents else 0.0
        action = remediations[0].get("action", "unknown") if remediations else "unknown"
        target = remediations[0].get("target", svc) if remediations else svc
        conf = remediations[0].get("confidence", 0.0) if remediations else 0.0
        hist = remediations[0].get("historical_outcome", "unknown") if remediations else "unknown"

        return (
            f"Incident on {svc} triggered by {trigger}. "
            f"Analyzed {n_events} telemetry signals in the preceding window. "
            f"Found {n_matches} similar past incidents; "
            f"best match: {top_match} (similarity={top_sim:.3f}). "
            f"Recommended action: {action} on {target} "
            f"(confidence={conf:.2f}, historical outcome: {hist})."
        )
