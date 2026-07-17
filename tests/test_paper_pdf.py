from __future__ import annotations

from pathlib import Path


def test_paper_spec_builder_outputs_source_and_pdf() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "paper" / "HistoDelib_Paper_Writing_Spec_CN.md"
    pdf = root / "docs" / "paper" / "HistoDelib_Paper_Writing_Spec_CN.pdf"
    script = root / "scripts" / "build_paper_writing_spec.py"

    assert source.exists()
    assert pdf.exists()
    assert script.exists()
    text = source.read_text(encoding="utf-8")
    for required in ("任务定义", "模态隔离", "消融实验", "评价指标", "NOT_RUN", "N/A"):
        assert required in text
    assert "虚构实验结果" not in text
