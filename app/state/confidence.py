"""Confidence-bearing estimates.

Every model-derived number in BabyMotion is wrapped in ``Estimate`` with a
confidence and human-readable evidence, so downstream code can refuse to act
on weak signals (confidence < min_confidence -> no strong response).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Estimate:
    value: float
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.value = float(min(1.0, max(0.0, self.value)))
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    @property
    def reliable(self) -> bool:
        return self.confidence >= 0.4

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 3),
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


def combine(*values: float) -> float:
    """Multiplicative combination of independent confidence factors."""
    out = 1.0
    for v in values:
        out *= min(1.0, max(0.0, v))
    return out
