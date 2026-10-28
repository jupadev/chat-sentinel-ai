import pytest

from sentinel.config import Thresholds
from sentinel.llm import LlmVerdict
from sentinel.models import Decision, RuleHit
from sentinel.pipeline import Moderator, decide

T = Thresholds(llm_min=0.4, llm_max=0.8, hide=0.98)


@pytest.mark.parametrize(
    "risk, verdict, doubtful",
    [(0.1, "ok", False), (0.5, "flag", True), (0.9, "flag", False), (0.99, "hide", False)],
)
def test_decide_by_risk_band(risk, verdict, doubtful):
    d = decide([], risk, {"toxicity": risk}, T)
    assert (d.verdict, d.doubtful) == (verdict, doubtful)


def test_acting_rule_wins_over_low_risk_and_ignore_hits_do_not_act():
    hide = RuleHit("slurs", "hide", "hate", "sudaca")
    mild = RuleHit("mild_insults", "ignore", "insult", "boludo")
    assert decide([mild, hide], 0.05, {}, T).verdict == "hide"
    assert decide([mild], 0.05, {}, T).verdict == "ok"


class FakeEncoders:
    pass


def doubtful() -> Decision:
    return decide([], 0.5, {"toxicity": 0.5}, T)


@pytest.mark.parametrize(
    "llm, can_hide, verdict, stage",
    [
        (LlmVerdict("ok", "none", "trato amistoso"), False, "ok", "llm"),
        (LlmVerdict("hide", "threat", "amenaza"), False, "flag", "llm"),
        (LlmVerdict("hide", "threat", "amenaza"), True, "hide", "llm"),
        (RuntimeError("ollama down"), False, "flag", "encoders"),
    ],
)
def test_adjudicate(config, message, llm, can_hide, verdict, stage):
    def judge(*_):
        if isinstance(llm, Exception):
            raise llm
        return llm

    cfg = config.model_copy(update={"llm": config.llm.model_copy(update={"enabled": True, "can_hide": can_hide})})

    class History:
        def conversation_context(self, msg, limit):
            return []

    d = Moderator(cfg, FakeEncoders(), judge).adjudicate(message("che"), doubtful(), History())
    assert (d.verdict, d.stage) == (verdict, stage)
