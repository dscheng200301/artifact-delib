from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_deferred_paper_prompt_forbids_research_results() -> None:
    prompt = (ROOT / "docs" / "prompts" / "generate_paper_writing_spec_pdf.md").read_text(
        encoding="utf-8"
    )

    for required in ("任务定义", "方法章节", "消融实验", "评价指标", "NOT_RUN", "N/A"):
        assert required in prompt
    assert "不得编造任何实验结果" in prompt
