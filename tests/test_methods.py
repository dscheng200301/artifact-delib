from __future__ import annotations

from pathlib import Path

from histodelib.api.mock import MockModelClient
from histodelib.data.fixture_builder import build_fixture
from histodelib.methods.agents import ImageAgent, TextAgent
from histodelib.methods.baselines import BASELINE_NAMES, create_baseline
from histodelib.methods.cross_exam import ControlledCrossExamination
from histodelib.methods.histodelib import HistoDelibMethod
from histodelib.methods.judge import DeferredJudge
from histodelib.methods.probe import LightRelationProbe
from histodelib.methods.reinspection import TargetedReinspection
from histodelib.methods.router import ApiRouter, RuleRouter
from histodelib.prompts.loader import load_prompt
from histodelib.schemas import Label


def test_rule_router_only_escalates_when_probe_has_risk() -> None:
    router = RuleRouter()

    assert router.route({"risk_flags": []}).reinspect is False
    escalated = router.route({"risk_flags": ["temporal_conflict", "unreadable_glyph"]})
    assert escalated.reinspect is True
    assert escalated.reinspection_targets == ("text", "glyph")


def test_api_router_returns_schema_validated_route() -> None:
    router = ApiRouter(client=MockModelClient(role="router"))

    decision = router.route({"risk_flags": ["modality_disagreement"]})

    assert decision.reinspect is True
    assert decision.reason.startswith("api:")


def test_histodelib_api_router_counts_router_call(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]
    method = HistoDelibMethod(
        client=MockModelClient(role="vlm"),
        router=ApiRouter(client=MockModelClient(role="router")),
        method_name="histodelib_api_router",
    )

    prediction = method.run(sample)

    assert prediction.api_calls == 3


def test_histodelib_api_router_call_count_is_per_sample(tmp_path: Path) -> None:
    samples = build_fixture(tmp_path)
    router = ApiRouter(client=MockModelClient(role="router"))
    method = HistoDelibMethod(
        client=MockModelClient(role="vlm"), router=router, method_name="histodelib_api_router"
    )

    predictions = [method.run(sample) for sample in samples[:2]]

    assert [prediction.api_calls for prediction in predictions] == [3, 3]


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
    sample = next(
        sample for sample in build_fixture(tmp_path) if sample.label is Label.MISCAPTIONED
    )
    method = HistoDelibMethod(client=MockModelClient(role="vlm"), router=RuleRouter())

    prediction = method.run(sample)

    assert prediction.final_label is Label.MISCAPTIONED
    assert prediction.status == "COMPLETED"
    assert prediction.api_calls >= 1
    assert "probe_flags" in prediction.evidence
    assert "cross_exam_rounds" in prediction.evidence


def test_text_and_image_agents_keep_inputs_isolated(tmp_path: Path) -> None:
    class RecordingClient(MockModelClient):
        def __init__(self) -> None:
            super().__init__(role="agent")
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    image_path = build_fixture(tmp_path)[0].image_path
    client = RecordingClient()
    prompt_root = Path(__file__).parents[1] / "prompts"
    TextAgent(client, prompt=load_prompt(prompt_root / "text_agent" / "v1.yaml")).analyze(
        "caption only"
    )
    ImageAgent(client, prompt=load_prompt(prompt_root / "image_agent" / "v1.yaml")).analyze(
        image_path
    )

    assert client.requests[0].image_base64 is None
    assert "caption only" in client.requests[0].user_prompt
    assert client.requests[1].image_base64 is not None
    assert "caption only" not in client.requests[1].user_prompt


def test_probe_reinspection_and_cross_exam_are_bounded() -> None:
    probe = LightRelationProbe().assess(
        {"label": "TRUE", "claims": ["1912"]},
        {"label": "MISCAPTIONED", "visible_text": "1912"},
    )
    decision = TargetedReinspection().select(probe)
    result = ControlledCrossExamination(max_rounds=2).run(probe, decision)

    assert "modality_disagreement" in probe.risk_flags
    assert decision.targets
    assert result.rounds <= 2
    assert result.stop_reason in {"stable", "max_rounds", "abstain"}


def test_baseline_factory_exposes_named_api_only_protocols(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    for name in BASELINE_NAMES:
        prediction = create_baseline(name, MockModelClient(role="baseline")).run(sample)
        assert prediction.method == name
        assert prediction.final_label is not None

    assert (
        create_baseline("direct_vlm", MockModelClient(role="baseline")).run(sample).api_calls == 1
    )
    assert (
        create_baseline("self_consistency", MockModelClient(role="baseline")).run(sample).api_calls
        == 3
    )
    assert (
        create_baseline("generic_mad", MockModelClient(role="baseline")).run(sample).api_calls == 4
    )
