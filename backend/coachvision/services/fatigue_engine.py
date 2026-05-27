"""Rule-based fatigue / readiness engine (v1) with explainability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fatigue_features import RollingFeatureSnapshot, snapshot_to_dict

LOAD_BEARING_EXERCISES = frozenset(
    {
        "squat",
        "lunge",
        "deadlift",
        "bicep_curl",
        "shoulder_press",
    }
)
STATIC_HOLD_EXERCISES = frozenset({"plank", "wall_sit"})


@dataclass
class ExplainabilityFactor:
    key: str
    label: str
    impact: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "impact": self.impact,
            "detail": self.detail,
        }


@dataclass
class FatiguePredictResult:
    readiness_score: int
    fatigue_level: str
    recommendation: str
    factors: list[str]
    explainability: list[ExplainabilityFactor] = field(default_factory=list)
    feature_snapshot: dict[str, Any] = field(default_factory=dict)


def _parse_user_context(ctx: dict[str, Any]) -> tuple[float, int, int, float, float | None, float | None]:
    sleep = float(ctx.get("sleepHours", ctx.get("sleep_hours", 7.5)))
    soreness = int(ctx.get("muscleSoreness", ctx.get("soreness", ctx.get("muscle_soreness", 2))))
    stress = int(ctx.get("stress", 2))
    external_load = float(ctx.get("externalLoadKg", ctx.get("external_load_kg", ctx.get("loadKg", 0))) or 0)
    body_weight_raw = ctx.get("bodyWeightKg", ctx.get("body_weight_kg"))
    body_weight = float(body_weight_raw) if body_weight_raw not in (None, "") else None
    external_load = max(0.0, min(300.0, external_load))
    if body_weight is not None:
        body_weight = max(20.0, min(400.0, body_weight))
    load_ratio = external_load / body_weight if body_weight and body_weight > 0 else None
    soreness = max(1, min(5, soreness))
    stress = max(1, min(5, stress))
    return sleep, soreness, stress, external_load, body_weight, load_ratio


def predict_rule_v1(
    rolling: RollingFeatureSnapshot,
    user_context: dict[str, Any],
    *,
    window_days: int = 7,
    exercise_id: str | None = None,
) -> FatiguePredictResult:
    """
    Start from a neutral base and apply transparent adjustments.
    Negative impact = lowers readiness score.
    """
    normalized_exercise = str(exercise_id or "").strip().lower().replace(" ", "_").replace("-", "_")
    sleep, soreness, stress, external_load, body_weight, load_ratio = _parse_user_context(user_context or {})
    is_hold_exercise = normalized_exercise in STATIC_HOLD_EXERCISES
    volume_unit = "seconds" if is_hold_exercise else "reps"
    if normalized_exercise not in LOAD_BEARING_EXERCISES:
        external_load = 0.0
        load_ratio = None
    explain: list[ExplainabilityFactor] = []
    score = 78

    # --- Volume (recent sessions + reps) ---
    vol_penalty = min(24, rolling.sessions_7d * 3)
    if vol_penalty:
        explain.append(
            ExplainabilityFactor(
                key="session_volume_7d",
                label="Recent session frequency",
                impact=-vol_penalty,
                detail=f"{rolling.sessions_7d} completed session(s) in the last {window_days} days.",
            )
        )
        score -= vol_penalty

    extra_reps = max(0, rolling.reps_7d - 100)
    rep_penalty = int(min(18, extra_reps * 0.12))
    if rep_penalty:
        explain.append(
            ExplainabilityFactor(
                key="rep_volume_7d",
                label="Recent hold time" if is_hold_exercise else "Recent rep volume",
                impact=-rep_penalty,
                detail=f"{rolling.reps_7d} total {volume_unit} in-window; high volume adds fatigue load.",
            )
        )
        score -= rep_penalty

    # --- Form / quality degradation ---
    if rolling.avg_form_score_7d is not None and rolling.avg_form_score_7d < 82:
        fp = int(min(14, round((82 - rolling.avg_form_score_7d) * 0.6)))
        explain.append(
            ExplainabilityFactor(
                key="avg_form_7d",
                label="Recent form quality",
                impact=-fp,
                detail=f"Average form score over the window is {rolling.avg_form_score_7d:.1f}/100.",
            )
        )
        score -= fp

    if rolling.form_degradation_7d is not None and rolling.form_degradation_7d < -6:
        explain.append(
            ExplainabilityFactor(
                key="form_trend_7d",
                label="Form trend",
                impact=-10,
                detail=f"Later sessions score lower than earlier ones (Δ ≈ {rolling.form_degradation_7d:.1f}).",
            )
        )
        score -= 10

    if rolling.shallow_rep_ratio_7d is not None and rolling.shallow_rep_ratio_7d > 0.35:
        explain.append(
            ExplainabilityFactor(
                key="shallow_rep_ratio",
                label="Depth / quality reps",
                impact=-8,
                detail=f"{round(100 * rolling.shallow_rep_ratio_7d)}% of tracked reps were shallow or very shallow.",
            )
        )
        score -= 8

    # --- ROM trend (proxy for mechanical fatigue) ---
    if rolling.rom_trend_7d is not None and rolling.rom_trend_7d < -4:
        explain.append(
            ExplainabilityFactor(
                key="rom_trend_7d",
                label="Range-of-motion trend",
                impact=-10,
                detail=f"ROM across sessions is trending down (Δ ≈ {rolling.rom_trend_7d:.1f}°).",
            )
        )
        score -= 10

    # --- Recovery spacing ---
    if rolling.days_since_last_session is None:
        explain.append(
            ExplainabilityFactor(
                key="no_prior_session",
                label="Training history",
                impact=-4,
                detail="No earlier completed session found for this exercise; assuming unfamiliar load.",
            )
        )
        score -= 4
    elif rolling.days_since_last_session <= 0:
        explain.append(
            ExplainabilityFactor(
                key="same_day_repeat",
                label="Recovery spacing",
                impact=-10,
                detail="Another session for this exercise ended very recently.",
            )
        )
        score -= 10
    elif rolling.days_since_last_session >= 2:
        bonus = min(8, rolling.days_since_last_session * 2)
        explain.append(
            ExplainabilityFactor(
                key="recovery_days",
                label="Recovery spacing",
                impact=bonus,
                detail=f"About {rolling.days_since_last_session} day(s) since the last session for this lift.",
            )
        )
        score += bonus

    # --- Planned external load (dumbbells / weighted squat) ---
    if external_load > 0 and normalized_exercise in LOAD_BEARING_EXERCISES:
        if load_ratio is not None:
            load_penalty = int(min(20, max(2, round(load_ratio * 42))))
            detail = (
                f"Planned external load is {external_load:.1f}kg "
                f"(about {round(load_ratio * 100)}% of bodyweight)."
            )
        else:
            load_penalty = int(min(16, max(2, round(external_load * 0.45))))
            detail = (
                f"Planned external load is {external_load:.1f}kg; bodyweight was not available, "
                "so the load penalty is estimated from absolute weight."
            )
        explain.append(
            ExplainabilityFactor(
                key="external_load",
                label="Added exercise load",
                impact=-load_penalty,
                detail=detail,
            )
        )
        score -= load_penalty

    # --- Self-reported readiness ---
    if sleep < 6:
        p = 16
        explain.append(
            ExplainabilityFactor(
                key="sleep",
                label="Sleep",
                impact=-p,
                detail=f"Reported sleep ~{sleep:.1f}h (< 6h) strongly reduces readiness.",
            )
        )
        score -= p
    elif sleep < 7:
        p = 8
        explain.append(
            ExplainabilityFactor(
                key="sleep",
                label="Sleep",
                impact=-p,
                detail=f"Reported sleep ~{sleep:.1f}h is below the usual 7h target.",
            )
        )
        score -= p

    if soreness >= 4:
        p = 12
        explain.append(
            ExplainabilityFactor(
                key="soreness",
                label="Muscle soreness",
                impact=-p,
                detail=f"Reported soreness {soreness}/5 suggests incomplete recovery.",
            )
        )
        score -= p
    elif soreness == 3:
        explain.append(
            ExplainabilityFactor(
                key="soreness",
                label="Muscle soreness",
                impact=-6,
                detail="Moderate soreness (3/5) slightly reduces readiness.",
            )
        )
        score -= 6

    if stress >= 4:
        explain.append(
            ExplainabilityFactor(
                key="stress",
                label="Stress",
                impact=-10,
                detail=f"Reported stress {stress}/5 adds central fatigue risk.",
            )
        )
        score -= 10
    elif stress == 3:
        explain.append(
            ExplainabilityFactor(
                key="stress",
                label="Stress",
                impact=-5,
                detail="Elevated stress (3/5) has a small negative effect on readiness.",
            )
        )
        score -= 5

    score = max(0, min(100, int(round(score))))

    if score >= 72:
        level = "low"
        recommendation = (
            "Readiness looks solid. You can run your planned session; keep usual rest and technique cues."
        )
    elif score >= 48:
        level = "moderate"
        recommendation = (
            "Reduce planned sets or reps by about 10–15%, extend rest slightly, and stop if form degrades."
        )
    else:
        level = "high"
        recommendation = (
            "Favor a light or mobility-focused day, or switch to an easier variation until recovery improves."
        )

    factors = [f"{e.label}: {e.detail} ({e.impact:+d})" for e in explain]
    snap = snapshot_to_dict(rolling)
    snap["window_days"] = window_days
    snap["self_report"] = {
        "sleepHours": sleep,
        "muscleSoreness": soreness,
        "stress": stress,
        "externalLoadKg": external_load,
        "bodyWeightKg": body_weight,
        "externalLoadBodyweightRatio": round(load_ratio, 3) if load_ratio is not None else None,
    }

    return FatiguePredictResult(
        readiness_score=score,
        fatigue_level=level,
        recommendation=recommendation,
        factors=factors,
        explainability=explain,
        feature_snapshot=snap,
    )
