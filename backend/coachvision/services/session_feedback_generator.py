"""Rule-based v1 session feedback from live workout export data (rep_metrics + summary)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from ..db.models import RepEvent
from ..db.models import Session as WorkoutSession
from ..db.models import SessionFeedback

FEEDBACK_VERSION = "feedback_rule_v1"

_SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}

_ISSUE_META: dict[str, dict[str, str]] = {
    "depth_shallow": {
        "label": "Partial depth",
        "severity": "medium",
        "why": "Some reps stopped short of full range.",
        "fix": "Focus on reaching consistent depth each rep.",
    },
    "depth_very_shallow": {
        "label": "Very shallow reps",
        "severity": "high",
        "why": "Several reps used minimal range of motion.",
        "fix": "Lower under control until you hit your depth target.",
    },
    "squat_depth_shallow": {
        "label": "Squat depth was slightly shallow",
        "severity": "medium",
        "why": "One or more squat reps did not reach the target bottom position.",
        "fix": "Sit the hips back and lower until the knees reach the target angle before standing.",
    },
    "squat_depth_very_shallow": {
        "label": "Squats were too shallow",
        "severity": "high",
        "why": "Some reps used too little knee bend, so the movement range was incomplete.",
        "fix": "Slow the descent and aim to reach parallel depth before driving up.",
    },
    "squat_limited_rom": {
        "label": "Limited squat range of motion",
        "severity": "medium",
        "why": "The difference between the top and bottom knee angle was small.",
        "fix": "Start tall, descend deeper, then return fully to standing on every rep.",
    },
    "squat_rushed_rep": {
        "label": "Rushed squat tempo",
        "severity": "low",
        "why": "Some reps were completed very quickly, which can reduce control and form quality.",
        "fix": "Use a controlled rhythm: lower smoothly, pause briefly, then stand up.",
    },
    "squat_inconsistent_depth": {
        "label": "Inconsistent squat depth",
        "severity": "medium",
        "why": "The bottom position changed noticeably between reps.",
        "fix": "Pick one depth target and try to hit the same bottom position each rep.",
    },
    "squat_incomplete_stand": {
        "label": "Incomplete standing position",
        "severity": "medium",
        "why": "Some reps did not clearly return to a tall standing position.",
        "fix": "Finish each rep by standing tall before starting the next descent.",
    },
}


def _slug_issue(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:48] or "form_issue") if s else "form_issue"


def _severity_for_code(code: str) -> str:
    if code in _ISSUE_META:
        return _ISSUE_META[code].get("severity", "low")
    if code in ("depth_very_shallow",):
        return "high"
    if code in ("depth_shallow",):
        return "medium"
    return "low"


def _normalize_exercise_id(exercise_id: str | None) -> str:
    return str(exercise_id or "").strip().lower().replace(" ", "_").replace("-", "_")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):  # noqa: PLR0124
        return None
    return x


def _duration_seconds(rep: dict[str, Any]) -> float | None:
    seconds = _safe_float(rep.get("duration"))
    if seconds is not None:
        return seconds
    millis = _safe_float(rep.get("duration_ms"))
    if millis is not None:
        return millis / 1000.0
    return None


def _depth_issue_code(exercise_id: str, depth_quality: str) -> str | None:
    if depth_quality == "shallow":
        return "squat_depth_shallow" if exercise_id == "squat" else "depth_shallow"
    if depth_quality == "very_shallow":
        return "squat_depth_very_shallow" if exercise_id == "squat" else "depth_very_shallow"
    return None


def _meta_for_code(code: str) -> dict[str, str]:
    if code in _ISSUE_META:
        return _ISSUE_META[code]
    return {
        "label": code.replace("_", " ").title(),
        "severity": _severity_for_code(code),
        "why": "Observed across one or more reps.",
        "fix": "Review the movement and adjust your form next set.",
    }


def _add_rep_metric_issues(exercise_id: str, rep: dict[str, Any], add_issue) -> None:
    if exercise_id != "squat":
        return

    depth_quality = rep.get("depth_quality")
    rom = _safe_float(rep.get("range_of_motion"))
    if rom is not None and rom < 55 and depth_quality not in ("shallow", "very_shallow"):
        add_issue("squat_limited_rom")

    duration = _duration_seconds(rep)
    if duration is not None and duration < 0.75:
        add_issue("squat_rushed_rep")

    max_angle = _safe_float(rep.get("max_angle"))
    if max_angle is not None and max_angle < 160:
        add_issue("squat_incomplete_stand")


def export_data_from_rep_events(
    _workout: WorkoutSession,
    reps: list[RepEvent],
) -> dict[str, Any]:
    """Rebuild exporter-shaped payload from persisted rep_events (REST or hybrid flows)."""
    rep_metrics: list[dict[str, Any]] = []
    good = 0
    for r in sorted(reps, key=lambda x: x.rep_number):
        dq = r.depth_quality
        if dq == "good":
            good += 1
        entry: dict[str, Any] = {
            "rep_number": r.rep_number,
            "depth_quality": dq,
            "form_score": float(r.form_score) if r.form_score is not None else None,
            "min_angle": float(r.min_angle) if r.min_angle is not None else None,
            "max_angle": float(r.max_angle) if r.max_angle is not None else None,
            "range_of_motion": float(r.range_of_motion) if r.range_of_motion is not None else None,
            "duration": float(r.duration_ms) / 1000.0 if r.duration_ms is not None else None,
        }
        extra = r.extra if isinstance(r.extra, dict) else {}
        fi = extra.get("form_issues")
        if isinstance(fi, list):
            entry["form_issues"] = fi
        rep_metrics.append(entry)

    total = len(rep_metrics)
    summary: dict[str, Any] = {"total_reps": total, "good_reps": good}
    return {"summary": summary, "rep_metrics": rep_metrics}


def export_data_minimal(workout: WorkoutSession) -> dict[str, Any]:
    """When no per-rep data exists — e.g. session ended via REST only."""
    tr = int(workout.total_reps or 0)
    summary: dict[str, Any] = {"total_reps": tr}
    if workout.avg_form_score is not None and tr > 0:
        summary["good_reps"] = int(round(tr * float(workout.avg_form_score) / 100.0))
    return {"summary": summary, "rep_metrics": []}


def ensure_session_feedback_for_completed_workout(
    db: DbSession,
    workout: WorkoutSession,
    *,
    now: datetime | None = None,
) -> None:
    """
    Insert session_feedback if missing (e.g. POST /sessions/{id}/end).
    Skips when a row already exists (e.g. live WS finalize already wrote one).
    """
    existing = db.scalars(
        select(SessionFeedback.id).where(SessionFeedback.session_id == workout.id).limit(1)
    ).first()
    if existing is not None:
        return

    reps = db.scalars(
        select(RepEvent).where(RepEvent.session_id == workout.id).order_by(RepEvent.rep_number)
    ).all()

    if reps:
        export_data = export_data_from_rep_events(workout, reps)
        src = "rep_events_v1"
    else:
        export_data = export_data_minimal(workout)
        src = "rest_completed_v1"

    ts = now or datetime.now(timezone.utc)
    row = build_session_feedback_row(workout, export_data, now=ts, export_source=src)
    db.add(row)


def build_session_feedback_row(
    workout: WorkoutSession,
    export_data: dict[str, Any],
    *,
    now: datetime | None = None,
    export_source: str = "live_export_v1",
) -> SessionFeedback:
    """Build a SessionFeedback row from dispatcher export_session_data()."""
    ts = now or datetime.now(timezone.utc)
    inner = export_data.get("summary") or {}
    rep_metrics = export_data.get("rep_metrics") or []
    exercise_id = _normalize_exercise_id(workout.exercise_id)
    measurement_type = str(inner.get("measurement_type") or "reps").lower()
    total_reps = int(inner.get("total_reps", 0))
    if total_reps <= 0 and rep_metrics:
        total_reps = len(rep_metrics)

    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    squat_min_angles: list[float] = []

    def add_issue(code: str, label: str | None = None, n: int = 1) -> None:
        counts[code] = counts.get(code, 0) + n
        labels.setdefault(code, label or _meta_for_code(code)["label"])

    for rep in rep_metrics:
        if not isinstance(rep, dict):
            continue
        dq = rep.get("depth_quality")
        if isinstance(dq, str):
            depth_code = _depth_issue_code(exercise_id, dq)
            if depth_code:
                add_issue(depth_code)

        min_angle = _safe_float(rep.get("min_angle"))
        if exercise_id == "squat" and min_angle is not None:
            squat_min_angles.append(min_angle)

        _add_rep_metric_issues(exercise_id, rep, add_issue)

        raw_issues = rep.get("form_issues")
        if isinstance(raw_issues, list):
            for w in raw_issues:
                if isinstance(w, str) and w.strip():
                    code = _slug_issue(w)
                    add_issue(code, w.strip()[:120])

    if exercise_id == "squat" and len(squat_min_angles) >= 3:
        if max(squat_min_angles) - min(squat_min_angles) >= 18:
            add_issue("squat_inconsistent_depth")

    errors_count = sum(counts.values())

    rating = 0
    if workout.avg_form_score is not None:
        rating = int(round(float(workout.avg_form_score)))
    elif total_reps > 0:
        good = inner.get("good_reps")
        if good is not None:
            rating = int(round(100.0 * int(good) / total_reps))

    sorted_codes = sorted(
        counts.keys(),
        key=lambda c: (-counts[c], -_SEVERITY_RANK.get(_severity_for_code(c), 0)),
    )
    top_errors: list[dict[str, Any]] = []
    for code in sorted_codes[:8]:
        meta = _meta_for_code(code)
        top_errors.append(
            {
                "code": code,
                "label": labels.get(code, meta["label"]),
                "count": counts[code],
                "severity": meta["severity"],
            }
        )

    error_breakdown: dict[str, Any] = {}
    for code, n in counts.items():
        meta = _meta_for_code(code)
        entry: dict[str, Any] = {
            "count": n,
            "severity": meta["severity"],
            "why": meta["why"],
            "fix": meta["fix"],
        }
        if code not in _ISSUE_META:
            entry["why"] = labels.get(code, meta["why"])
        error_breakdown[code] = entry

    action_items: list[dict[str, Any]] = []
    prio = 1
    for code in sorted_codes[:6]:
        title = labels.get(code, _meta_for_code(code)["label"])
        eb = error_breakdown.get(code, {})
        action_items.append(
            {
                "priority": prio,
                "title": title[:160],
                "why": eb.get("why", ""),
                "how_to_fix": eb.get("fix", ""),
                "cue": eb.get("fix", "")[:80],
            }
        )
        prio += 1

    dq_pct = inner.get("detection_quality")
    conf: float | None = None
    if dq_pct is not None:
        try:
            conf = max(0.0, min(1.0, float(dq_pct) / 100.0))
        except (TypeError, ValueError):
            conf = None
    elif total_reps > 0:
        conf = 0.82

    signals_used = {
        "total_reps": total_reps,
        "errors_count": errors_count,
        "avg_form_score": workout.avg_form_score,
        "good_reps": inner.get("good_reps"),
        "measurement_type": measurement_type,
        "total_seconds": inner.get("total_seconds"),
        "source": export_source,
    }

    summary_text = _compose_summary(total_reps, rating, counts, sorted_codes, labels, measurement_type)

    return SessionFeedback(
        id=uuid4(),
        user_id=workout.user_id,
        exercise_id=workout.exercise_id,
        session_id=workout.id,
        overall_rating=max(0, min(100, rating)),
        summary_text=summary_text,
        errors_count=errors_count,
        top_errors=top_errors,
        error_breakdown=error_breakdown,
        action_items=action_items,
        confidence_overall=conf,
        signals_used=signals_used,
        version=FEEDBACK_VERSION,
        generated_at=ts,
        updated_at=ts,
    )


def _compose_summary(
    total_reps: int,
    rating: int,
    counts: dict[str, int],
    sorted_codes: list[str],
    labels: dict[str, str],
    measurement_type: str = "reps",
) -> str:
    is_hold = measurement_type == "hold"
    unit = "second(s)" if is_hold else "rep(s)"
    if total_reps <= 0:
        if is_hold:
            return "No hold time was recorded for this session."
        return "No reps were recorded for this session."
    if not counts:
        if is_hold:
            return f"Solid hold - {total_reps} second(s) of tracked position. Keep building time under control."
        return f"Solid session — {total_reps} rep(s) with no major issues flagged. Keep it up."
    top = sorted_codes[0]
    label = labels.get(top, _meta_for_code(top)["label"])[:100]
    parts = [
        f"Completed {total_reps} {unit}; overall score around {rating}/100.",
        f"Main focus: {label.lower()} ({counts[top]} instance(s)).",
        "See action items for cues to improve next time.",
    ]
    return " ".join(parts)
