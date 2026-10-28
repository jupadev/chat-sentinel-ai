import logging
import time
from collections.abc import Callable

from sentinel.config import Config, Thresholds
from sentinel.encoders import EncoderSet
from sentinel.llm import LlmVerdict
from sentinel.models import SEVERITY, ChatMessage, Decision, History, RuleHit
from sentinel.rules import RuleEngine

log = logging.getLogger(__name__)

Judge = Callable[[ChatMessage, list[ChatMessage], list[RuleHit]], LlmVerdict]


def decide(hits: list[RuleHit], risk: float | None, scores: dict[str, float], t: Thresholds) -> Decision:
    """Combine rule hits and the encoder risk into a verdict. The LLM only sees `doubtful` cases."""
    acting = [h for h in hits if h.action != "ignore"]
    if acting:
        worst = max(acting, key=lambda h: SEVERITY[h.action])
        return Decision(worst.action, worst.category, f"rule {worst.rule}: {worst.match}", "rules", risk, scores, hits)
    if risk is None or risk < t.llm_min:
        return Decision("ok", None, "low risk", "encoders", risk, scores, hits)
    top = max(scores, key=scores.get) if scores else None
    if risk >= t.hide:
        return Decision("hide", top, f"very high risk ({top})", "encoders", risk, scores, hits)
    if risk > t.llm_max:
        return Decision("flag", top, f"high risk ({top})", "encoders", risk, scores, hits)
    # Doubtful: flag until the LLM (if enabled) says otherwise; when in doubt, flag, never hide or pass.
    return Decision("flag", top, f"doubtful ({top})", "encoders", risk, scores, hits, doubtful=True)


class Moderator:
    def __init__(self, config: Config, encoders: EncoderSet, judge: Judge | None):
        self._config = config
        self._rules = RuleEngine(config.rules)
        self._encoders = encoders
        self._judge = judge

    def screen(self, messages: list[ChatMessage], history: History) -> list[Decision]:
        """Rules and encoders over the whole batch."""
        started = time.perf_counter()
        hits = [self._rules.evaluate(m, history) for m in messages]
        rules_ms = (time.perf_counter() - started) * 1000

        # A hide rule already settles the message; skip the encoders for it.
        pending = [i for i, h in enumerate(hits) if not any(x.action == "hide" for x in h)]
        started = time.perf_counter()
        scored = self._encoders.score_batch([messages[i].text for i in pending])
        encoders_ms = (time.perf_counter() - started) * 1000
        by_index = dict(zip(pending, scored))

        decisions = []
        for i, message_hits in enumerate(hits):
            risk, scores = by_index.get(i, (None, {}))
            decision = decide(message_hits, risk, scores, self._config.thresholds)
            decision.latency_ms = {
                "rules": round(rules_ms / len(messages), 2),
                "encoders_batch": round(encoders_ms, 1),
                "batch_size": len(messages),
            }
            decisions.append(decision)
        return decisions

    def adjudicate(self, message: ChatMessage, decision: Decision, history: History) -> Decision:
        """Ask the LLM about a doubtful message. Any failure keeps the safe preliminary flag."""
        if self._judge is None or not self._config.llm.enabled:
            return decision
        context = history.conversation_context(message, self._config.context.messages)
        started = time.perf_counter()
        try:
            result = self._judge(message, context, decision.rule_hits)
        except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to flag
            log.warning("llm failed for %s: %s", message.key, exc)
            decision.reason = "doubtful, LLM unavailable"
            return decision
        finally:
            decision.latency_ms["llm"] = round((time.perf_counter() - started) * 1000, 1)

        verdict = result.verdict
        if verdict == "hide" and not self._config.llm.can_hide:
            verdict = "flag"
        decision.verdict = verdict
        decision.category = None if verdict == "ok" or result.category == "none" else result.category
        decision.reason = result.reason
        decision.stage = "llm"
        return decision
