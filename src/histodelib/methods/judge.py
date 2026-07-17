"""Deferred adjudication over concise structured evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from histodelib.schemas import Label


@dataclass(frozen=True)
class JudgeResult:
    decision: Literal["KEEP", "REVISE", "ABSTAIN"]
    final_label: Label | None


class DeferredJudge:
    """Keep a blind label unless concise evidence warrants a revision."""

    def adjudicate(self, blind_label: Label | None, evidence: dict[str, Any]) -> JudgeResult:
        text_label = self._label(evidence.get("text_label"))
        image_label = self._label(evidence.get("image_label"))
        if text_label is None and image_label is None:
            return JudgeResult("ABSTAIN", blind_label)
        if text_label is not None and text_label == image_label:
            if text_label == blind_label:
                return JudgeResult("KEEP", blind_label)
            return JudgeResult("REVISE", text_label)
        if text_label is not None and text_label is not Label.TRUE:
            return JudgeResult("REVISE", text_label)
        return JudgeResult("KEEP", blind_label)

    @staticmethod
    def _label(value: object) -> Label | None:
        try:
            return Label(str(value))
        except ValueError:
            return None
