from __future__ import annotations

from pathlib import Path

from histodelib.api.mock import MockModelClient
from histodelib.data.fixture_builder import build_fixture
from histodelib.methods.histodelib import HistoDelibMethod
from histodelib.methods.judge import DeferredJudge
from histodelib.methods.router import RuleRouter
from histodelib.schemas import Label


def test_rule_router_only_escalates_when_probe_has_risk() -> None:
    router = RuleRouter()

    assert router.route({"risk_flags": []}).reinspect is False
    escalated = router.route({"risk_flags": ["temporal_conflict", "unreadable_glyph"]})
    assert escalated.reinspect is True
    assert escalated.reinspection_targets == ("text", "glyph")


def test_deferred_judge_can_keep_or_revise_blind_label() -> None:
    judge = DeferredJudge()

    kept = judge.adjudicate(Label.TRUE, {"text_label": "TRUE", "image_label": "TRUE"})
    revised = judge.adjudicate(
        Label.TRUE,
        {"text_label": "MISCAPTIONED", "image_label": "MISCAPTIONED"},
    )

    assert (kept.decision, kept.final_label) == ("KEEP", Label.TRUE)
    assert (revised.decision, revised.final_label) == ("REVISE", Label.MISCAPTIONED)


def test_histodelib_rule_runs_on_synthetic_fixture(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[1]
    method = HistoDelibMethod(client=MockModelClient(role="vlm"), router=RuleRouter())

    prediction = method.run(sample)

    assert prediction.final_label is Label.MISCAPTIONED
    assert prediction.status == "COMPLETED"
    assert prediction.api_calls >= 1
