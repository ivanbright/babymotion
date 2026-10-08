"""Session metrics.

Every metric has an explicit definition (see docstrings). No composite
"developmental scores" - only measurable interaction statistics.
"""

from __future__ import annotations

import statistics as _stats
from typing import Iterable, Optional, Sequence


def _latencies(responses: Iterable) -> list[float]:
    out = []
    for r in responses:
        if isinstance(r, dict):
            out.append(float(r.get("latency_s", r.get("latency", 0.0))))
        else:
            out.append(float(getattr(r, "latency_s")))
    return out


def response_rate(responses: int, stimuli: int) -> Optional[float]:
    """response_rate = responses / opportunities(stimuli). None if no stimuli."""
    if stimuli <= 0:
        return None
    return min(1.0, responses / stimuli)


def median_response_latency_ms(responses: Iterable) -> Optional[float]:
    """median(response_time - stimulus_time) in milliseconds."""
    lat = _latencies(responses)
    if not lat:
        return None
    return float(_stats.median(lat) * 1000.0)


def mean_attention(series: Sequence[tuple[float, float]]) -> Optional[float]:
    """Time-weighted mean attention estimate over the session."""
    if not series:
        return None
    if len(series) == 1:
        return float(series[0][1])
    total_t = 0.0
    weighted = 0.0
    for i in range(1, len(series)):
        dt = series[i][0] - series[i - 1][0]
        if dt <= 0:
            continue
        # trapezoidal weighting
        weighted += 0.5 * (series[i][1] + series[i - 1][1]) * dt
        total_t += dt
    if total_t <= 0:
        return float(_stats.mean([v for _, v in series]))
    return weighted / total_t


def max_attention(series: Sequence[tuple[float, float]]) -> Optional[float]:
    if not series:
        return None
    return float(max(v for _, v in series))


def sustained_attention(series: Sequence[tuple[float, float]],
                        threshold: float = 0.65) -> float:
    """Seconds during which the attention estimate stayed above threshold."""
    if len(series) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(series)):
        dt = series[i][0] - series[i - 1][0]
        if dt <= 0:
            continue
        if series[i][1] >= threshold and series[i - 1][1] >= threshold:
            total += dt
        elif series[i - 1][1] >= threshold > series[i][1]:
            frac = (series[i - 1][1] - threshold) / max(1e-9, series[i - 1][1] - series[i][1])
            total += dt * frac
        elif series[i][1] >= threshold > series[i - 1][1]:
            frac = (series[i][1] - threshold) / max(1e-9, series[i][1] - series[i - 1][1])
            total += dt * (1.0 - frac)
    return total


def responses_per_minute(responses: int, duration_s: float) -> Optional[float]:
    """interaction_rate = baby_responses / minute of session."""
    if duration_s <= 0:
        return None
    return responses / (duration_s / 60.0)


def novelty_response_ratio(baseline: Sequence[float],
                           after_novelty: Sequence[float]) -> Optional[float]:
    """Novelty response = mean(attention after novel stimulus) / mean(baseline attention).

    >1.0 means attention rose following novelty. Requires >=1 sample each side.
    """
    if not baseline or not after_novelty:
        return None
    base = float(_stats.mean(baseline))
    after = float(_stats.mean(after_novelty))
    if base <= 1e-6:
        return None
    return after / base


def imitation_rate(matching_responses: int, opportunities: int) -> Optional[float]:
    """imitation_rate = successful matching responses / imitation opportunities.

    E.g. during the Follow activity: ORIENTATION_CHANGED responses to
    TURN_HEAD stimuli / total TURN_HEAD stimuli shown.
    """
    if opportunities <= 0:
        return None
    return min(1.0, matching_responses / opportunities)


def attention_stats(series: Sequence[tuple[float, float]]) -> dict:
    if not series:
        return {"mean": None, "max": None, "min": None, "samples": 0}
    values = [v for _, v in series]
    return {
        "mean": round(float(_stats.mean(values)), 3),
        "max": round(float(max(values)), 3),
        "min": round(float(min(values)), 3),
        "samples": len(values),
    }
