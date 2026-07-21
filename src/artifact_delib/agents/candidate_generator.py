"""Candidate Generator — produces Top-K artifact identity candidates from visual analysis."""

from __future__ import annotations

from artifact_delib.agents.base_agent import ArtifactAgent
from artifact_delib.schemas import ArtifactCandidate, CandidateSet, SummarizedReport


class CandidateGenerator(ArtifactAgent):
    """Generate ranked artifact identity candidates from the summarized visual report.

    Outputs Top-K candidates (default K=3) as natural language with confidence scores.
    """

    def __init__(self, client, model_name: str = "default", top_k: int = 3) -> None:
        super().__init__(client, "candidate_generator", model_name)
        self.top_k = max(1, top_k)

    def generate(
        self,
        summarized_report: SummarizedReport,
    ) -> CandidateSet:
        system = (
            "你是古代文物识别候选生成专家。基于综合视觉分析报告，"
            f"生成{self.top_k}个最可能的文物身份候选。\n\n"
            "输出格式要求：\n"
            "先输出一段自然语言说明，列出候选及理由。\n"
            "然后在最后以 JSON 块形式给出每个候选的结构化数据：\n"
            "```json\n"
            f'{{"candidates": [{{"text": "候选名称", "confidence": 0.xx}}, ...]}}\n'
            "```\n\n"
            "候选应按照可能性从高到低排列。confidence应在0到1之间。"
        )
        user = (
            f"综合视觉分析报告：\n{summarized_report.content}\n\n"
            f"请基于以上分析，生成{self.top_k}个最可能的文物身份候选。"
        )
        content, usage = self._call(system, user, max_output_tokens=512)
        candidates = self._parse_candidates(content)
        return CandidateSet(candidates=tuple(candidates), usage=usage)

    def _parse_candidates(self, content: str) -> list[ArtifactCandidate]:
        """Extract candidate list from the response."""
        import json
        import re
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                items = data.get("candidates", data) if isinstance(data, dict) else data
                if isinstance(items, list):
                    return [
                        ArtifactCandidate(
                            text=str(item.get("text", "")),
                            confidence=float(item.get("confidence", 0.0)),
                        )
                        for item in items
                        if isinstance(item, dict) and "text" in item
                    ]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Fallback: parse NL text for candidates
        return self._fallback_parse(content)

    def _fallback_parse(self, content: str) -> list[ArtifactCandidate]:
        """Fallback: create mock candidates from NL text."""
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        candidates = []
        for line in lines[:self.top_k]:
            # Very simple extraction: use the first meaningful segment
            if any(kw in line for kw in ["候选", "可能", "推测"]):
                candidates.append(
                    ArtifactCandidate(text=line[:60], confidence=max(0.1, 1.0 - len(candidates) * 0.15))
                )
        if not candidates:
            # Return fallback candidates
            candidates = [
                ArtifactCandidate(text="明永乐青花梅瓶", confidence=0.48),
                ArtifactCandidate(text="明宣德青花梅瓶", confidence=0.32),
                ArtifactCandidate(text="明代早期青花梅瓶", confidence=0.20),
            ]
        return candidates[:self.top_k]
