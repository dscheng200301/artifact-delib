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
from histodelib.schemas import Label, ModelResponse, TokenUsage


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
    assert decision.source == "fallback"
    assert "ROUTER_PARSE_FAILURE" in decision.reason_codes


def test_api_router_uses_valid_api_decision_and_preserves_source() -> None:
    class RoutingClient:
        def generate(self, request):
            return ModelResponse(
                request_id=request.request_id,
                content='{"action":"ACCEPT","targets":[],"confidence":0.9,"reason_codes":["stable"]}',
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                provider="fake",
                model=request.model,
            )

    decision = ApiRouter(RoutingClient()).route({"risk_flags": []})

    assert decision.action == "ACCEPT"
    assert decision.source == "api"
    assert decision.reinspect is False


def test_api_router_records_parse_failure_and_falls_back() -> None:
    class InvalidRoutingClient:
        def generate(self, request):
            return ModelResponse(
                request_id=request.request_id,
                content='{"action":"REINSPECT","targets":["not-allowed"]}',
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                provider="fake",
                model=request.model,
            )

    decision = ApiRouter(InvalidRoutingClient()).route(
        {"risk_flags": ["modality_disagreement"]}
    )

    assert decision.source == "fallback"
    assert "ROUTER_PARSE_FAILURE" in decision.reason_codes
    assert decision.reinspect is True


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


def test_histodelib_api_deliberation_adds_bounded_followup_calls(tmp_path: Path) -> None:
    sample = next(
        sample for sample in build_fixture(tmp_path) if sample.label is Label.MISCAPTIONED
    )
    method = HistoDelibMethod(
        client=MockModelClient(role="vlm"),
        router=RuleRouter(),
        enable_api_deliberation=True,
        max_reinspection_targets=1,
        max_cross_exam_rounds=1,
        model_name="custom-model",
    )

    prediction = method.run(sample)

    assert prediction.api_calls >= 4
    assert prediction.evidence["reinspection_api_calls"] <= 1
    assert prediction.evidence["cross_exam_api_calls"] <= 1
    assert prediction.evidence["judge_api_calls"] == 1


def test_histodelib_api_router_full_protocol_is_bounded(tmp_path: Path) -> None:
    sample = next(
        sample for sample in build_fixture(tmp_path) if sample.label is Label.MISCAPTIONED
    )

    prediction = create_baseline(
        "histodelib_api_router",
        MockModelClient(role="full-protocol"),
        enable_api_deliberation=True,
        max_reinspection_targets=1,
        max_cross_exam_rounds=1,
    ).run(sample)

    assert prediction.status == "COMPLETED"
    assert prediction.api_calls >= 5
    assert prediction.evidence["reinspection_api_calls"] <= 1
    assert prediction.evidence["cross_exam_api_calls"] <= 1
    assert prediction.evidence["judge_api_calls"] == 1


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
    assert client.requests[0].response_schema == {"type": "json_object"}
    assert client.requests[0].prompt_name == "text_agent"
    assert client.requests[0].prompt_content_hash
    assert client.requests[0].model == "qwen3.5-flash-2026-02-23"
    assert "caption only" in client.requests[0].user_prompt
    assert client.requests[1].image_base64 is not None
    assert client.requests[1].response_schema == {"type": "json_object"}
    assert client.requests[1].model == "qwen3.5-flash-2026-02-23"
    assert "caption only" not in client.requests[1].user_prompt


def test_agents_use_explicit_model_name(tmp_path: Path) -> None:
    image_path = build_fixture(tmp_path)[0].image_path
    class RecordingClient(MockModelClient):
        def __init__(self) -> None:
            super().__init__(role="agent")
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    client = RecordingClient()

    TextAgent(client, model_name="custom-model").analyze("caption")
    ImageAgent(client, model_name="custom-model").analyze(image_path)

    assert [request.model for request in client.requests] == ["custom-model", "custom-model"]


def test_modal_agents_return_typed_single_modality_evidence(tmp_path: Path) -> None:
    image_path = build_fixture(tmp_path)[0].image_path
    client = MockModelClient(role="agent")

    text_result = TextAgent(client).analyze("A harbor in 1912")
    image_result = ImageAgent(client).analyze(image_path)

    assert text_result.modality == "text"
    assert text_result.structured.evidence_id
    assert text_result.structured.caption_claims == ("A harbor in 1912",)
    assert image_result.modality == "image"
    assert image_result.structured.evidence_id
    assert image_result.structured.visible_text is None


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


def test_probe_does_not_treat_missing_visible_text_as_unreadable_glyph() -> None:
    result = LightRelationProbe().assess(
        {"label": "TRUE", "claims": ["A harbor in 1912"]},
        {"label": "TRUE", "visible_text": None},
    )

    assert "unreadable_glyph" not in result.risk_flags


def test_probe_flags_unreadable_text_only_when_text_evidence_requires_it() -> None:
    result = LightRelationProbe().assess(
        {"label": "TRUE", "claims": ["A sign reads 'Harbor'"], "requires_visible_text": True},
        {"label": "TRUE", "visible_text": None},
    )

    assert "unreadable_glyph" in result.risk_flags


def test_reinspection_api_calls_are_bounded_and_image_only(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    class RecordingClient(MockModelClient):
        def __init__(self) -> None:
            super().__init__(role="reinspect")
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    client = RecordingClient()
    decision = TargetedReinspection().select(
        LightRelationProbe().assess(
            {"label": "TRUE", "claims": ["caption"]}, {"label": "MISCAPTIONED"}
        )
    )
    result = TargetedReinspection().inspect(
        client, sample, decision, model_name="custom-model", max_targets=1
    )

    assert result.api_calls == 1
    assert client.requests[0].image_base64 is not None
    assert client.requests[0].model == "custom-model"
    assert client.requests[0].response_schema == {"type": "json_object"}
    assert result.evidence[0]["view"]["reason_code"] == "target_unavailable_full_fallback"


def test_cross_exam_api_rounds_respect_maximum(tmp_path: Path) -> None:
    probe = LightRelationProbe().assess(
        {"label": "TRUE", "claims": ["caption"]}, {"label": "MISCAPTIONED"}
    )
    decision = TargetedReinspection().select(probe)

    result = ControlledCrossExamination(max_rounds=2).run_with_client(
        MockModelClient(role="critic"), probe, decision, model_name="custom-model"
    )

    assert 0 <= result.api_calls <= 2


def test_cross_exam_state_carries_prior_transcript_between_rounds(tmp_path: Path) -> None:
    probe = LightRelationProbe().assess(
        {"label": "TRUE", "claims": ["caption"]}, {"label": "MISCAPTIONED"}
    )
    decision = TargetedReinspection().select(probe)

    class RecordingClient(MockModelClient):
        def __init__(self) -> None:
            super().__init__(role="critic")
            self.prompts: list[str] = []

        def generate(self, request):
            self.prompts.append(request.user_prompt)
            return super().generate(request)

    client = RecordingClient()
    result = ControlledCrossExamination(max_rounds=2).run_with_client(
        client, probe, decision, model_name="custom-model"
    )

    assert result.state is not None
    assert len(result.state.rounds) == 2
    assert "prior_transcript" in client.prompts[1]


def test_deferred_judge_api_path_returns_structured_decision() -> None:
    result = DeferredJudge().adjudicate_with_client(
        MockModelClient(role="judge"),
        blind_label=Label.TRUE,
        evidence={"text_label": "MISCAPTIONED", "image_label": "MISCAPTIONED"},
        model_name="custom-model",
    )

    assert result.final_label is Label.MISCAPTIONED
    assert result.decision in {"KEEP", "REVISE", "ABSTAIN"}


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


def test_baseline_predictions_expose_protocol_provenance(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    prediction = create_baseline("self_reflection", MockModelClient(role="baseline")).run(sample)

    assert prediction.evidence["protocol"]["name"] == "self_reflection"
    assert "engineering_approximation" in prediction.evidence["protocol"]


def test_self_reflection_receives_prior_structured_answer(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    class RecordingClient(MockModelClient):
        def __init__(self) -> None:
            super().__init__(role="baseline")
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    client = RecordingClient()
    create_baseline("self_reflection", client).run(sample)

    assert len(client.requests) == 2
    assert "prior_answer=" in client.requests[1].user_prompt


def test_baseline_factory_passes_configured_model_name(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    class RecordingClient(MockModelClient):
        def __init__(self) -> None:
            super().__init__(role="baseline")
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    client = RecordingClient()
    create_baseline("direct_vlm", client, model_name="custom-model").run(sample)

    assert client.requests[0].model == "custom-model"


def test_baseline_protocols_use_declared_modalities(tmp_path: Path) -> None:
    sample = build_fixture(tmp_path)[0]

    class RecordingClient(MockModelClient):
        def __init__(self) -> None:
            super().__init__(role="baseline")
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return super().generate(request)

    text_client = RecordingClient()
    create_baseline("text_only", text_client).run(sample)
    assert text_client.requests[0].image_base64 is None

    image_client = RecordingClient()
    create_baseline("image_only", image_client).run(sample)
    assert image_client.requests[0].image_base64 is not None

    direct_client = RecordingClient()
    create_baseline("direct_vlm", direct_client).run(sample)
    assert direct_client.requests[0].image_base64 is not None
    assert sample.caption in direct_client.requests[0].user_prompt
    assert '"label"' in direct_client.requests[0].system_prompt
    assert "MISCAPTIONED" in direct_client.requests[0].system_prompt
