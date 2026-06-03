"""WebSocket live workout channel (v1 contract aligned with mobile team)."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session as DbSession

from coachvision.ai.counters.dispatcher import CONFIG_PRESETS, ExerciseType
from coachvision.core.security import decode_token
from coachvision.db.models import Session as WorkoutSession
from coachvision.db.session import get_db
from coachvision.realtime.contract import SCHEMA_VERSION, WsErrorCode
from coachvision.realtime.pipeline import (
    LiveSessionState,
    process_jpeg_frame,
    process_pose_landmarks,
)
from coachvision.services.ws_persistence import (
    abort_active_workout,
    finalize_completed_workout,
    load_or_create_workout_for_ws_start,
    normalize_exercise_id,
)

router = APIRouter()


def _j(payload: dict[str, Any]) -> dict[str, Any]:
    return {"schemaVersion": SCHEMA_VERSION, **payload}


def _optional_bounded_float(value: Any, *, min_value: float, max_value: float) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):  # noqa: PLR0124
        return None
    return max(min_value, min(max_value, parsed))


def _first_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


@router.websocket("/ws/live")
async def websocket_live(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
    db: DbSession = Depends(get_db),
) -> None:
    await websocket.accept()

    session_state: LiveSessionState | None = None
    db_workout: WorkoutSession | None = None

    try:
        try:
            payload = decode_token(token)
            if payload.get("token_type") != "access":
                await websocket.send_json(
                    _j(
                        {
                            "type": "error",
                            "sessionId": None,
                            "message": "Invalid token type",
                            "code": WsErrorCode.INVALID_TOKEN,
                        }
                    )
                )
                await websocket.close(code=4401)
                return
            user_uuid = UUID(str(payload["sub"]))
        except Exception as exc:  # noqa: BLE001
            await websocket.send_json(
                _j(
                    {
                        "type": "error",
                        "sessionId": None,
                        "message": str(exc),
                        "code": WsErrorCode.INVALID_TOKEN,
                    }
                )
            )
            await websocket.close(code=4401)
            return

        while True:
            raw = await websocket.receive_text()
            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    _j(
                        {
                            "type": "error",
                            "sessionId": session_state.session_id if session_state else None,
                            "message": "invalid_json",
                            "code": WsErrorCode.BAD_MESSAGE,
                        }
                    )
                )
                continue

            mtype = msg.get("type")

            if mtype == "start":
                if session_state is not None:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": "session_already_active",
                                "code": WsErrorCode.ALREADY_ACTIVE,
                            }
                        )
                    )
                    continue

                exercise_name = str(msg.get("exerciseName", "squat")).strip()
                difficulty = str(msg.get("difficulty", "intermediate")).strip().lower()
                target_sets = int(msg.get("targetSets", 1))
                target_reps = int(msg.get("targetReps", 1))
                readiness_context = msg.get("readinessContext")
                if not isinstance(readiness_context, dict):
                    readiness_context = {}
                load_value = _first_value(msg, "externalLoadKg", "external_load_kg", "loadKg")
                if load_value is None:
                    load_value = _first_value(
                        readiness_context,
                        "externalLoadKg",
                        "external_load_kg",
                        "loadKg",
                    )
                body_weight_value = _first_value(msg, "bodyWeightKg", "body_weight_kg")
                if body_weight_value is None:
                    body_weight_value = _first_value(
                        readiness_context,
                        "bodyWeightKg",
                        "body_weight_kg",
                    )
                external_load_kg = _optional_bounded_float(
                    load_value,
                    min_value=0,
                    max_value=300,
                )
                body_weight_kg = _optional_bounded_float(
                    body_weight_value,
                    min_value=20,
                    max_value=400,
                )
                rest_sid = msg.get("sessionId")
                rest_uuid: UUID | None = None
                if rest_sid is not None:
                    try:
                        rest_uuid = UUID(str(rest_sid))
                    except (ValueError, TypeError):
                        await websocket.send_json(
                            _j(
                                {
                                    "type": "error",
                                    "sessionId": None,
                                    "message": "invalid_sessionId",
                                    "code": WsErrorCode.BAD_MESSAGE,
                                }
                            )
                        )
                        continue

                try:
                    ExerciseType.from_string(exercise_name)
                except ValueError as exc:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": None,
                                "message": str(exc),
                                "code": WsErrorCode.UNSUPPORTED_EXERCISE,
                            }
                        )
                    )
                    continue

                if difficulty not in CONFIG_PRESETS:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": None,
                                "message": f"Invalid difficulty: {difficulty}",
                                "code": WsErrorCode.BAD_DIFFICULTY,
                            }
                        )
                    )
                    continue

                exercise_id = normalize_exercise_id(exercise_name)
                workout, err = load_or_create_workout_for_ws_start(
                    db,
                    user_id=user_uuid,
                    exercise_name=exercise_name,
                    difficulty=difficulty,
                    rest_session_id=rest_uuid,
                    target_sets=target_sets,
                    target_reps=target_reps,
                    external_load_kg=external_load_kg,
                    body_weight_kg=body_weight_kg,
                )
                if err or workout is None:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": None,
                                "message": err or "start_failed",
                                "code": err or WsErrorCode.START_FAILED,
                            }
                        )
                    )
                    continue

                db_workout = workout
                try:
                    session_state = LiveSessionState(
                        session_id=str(workout.id),
                        exercise_name=exercise_id,
                        difficulty=difficulty,
                    )
                except ValueError as exc:
                    abort_active_workout(db, workout)
                    db_workout = None
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": str(workout.id),
                                "message": str(exc),
                                "code": WsErrorCode.UNSUPPORTED_EXERCISE,
                            }
                        )
                    )
                    continue
                await websocket.send_json(_j({"type": "started", "sessionId": str(workout.id)}))
                continue

            if session_state is None:
                await websocket.send_json(
                    _j(
                        {
                            "type": "error",
                            "sessionId": None,
                            "message": "send_start_first",
                            "code": WsErrorCode.NO_SESSION,
                        }
                    )
                )
                continue

            if mtype == "pose":
                if msg.get("sessionId") != session_state.session_id:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": "session_mismatch",
                                "code": WsErrorCode.NO_SESSION,
                            }
                        )
                    )
                    continue

                timestamp_ms = msg.get("timestampMs")
                if not isinstance(timestamp_ms, int):
                    timestamp_ms = None

                client_inference_ms = msg.get("clientInferenceMs")
                if not isinstance(client_inference_ms, (int, float)):
                    client_inference_ms = None

                landmarks = msg.get("landmarks")
                result = process_pose_landmarks(
                    session_state,
                    landmarks,
                    timestamp_ms,
                    client_inference_ms,
                )

                if result["kind"] == "error":
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": result["message"],
                                "code": result["code"],
                            }
                        )
                    )
                    continue

                await websocket.send_json(
                    _j(
                        {
                            "type": "metrics",
                            "sessionId": result["session_id"],
                            "count": result["count"],
                            "rawCount": result.get("raw_count"),
                            "measurementType": result.get("measurement_type"),
                            "measurementLabel": result.get("measurement_label"),
                            "holdDurationSec": result.get("hold_duration_sec"),
                            "totalHoldTimeSec": result.get("total_hold_time_sec"),
                            "bestHoldSec": result.get("best_hold_sec"),
                            "completedHolds": result.get("completed_holds"),
                            "state": result["state"],
                            "angle": result["angle"],
                            "feedback": result["feedback"],
                            "progress": result["progress"],
                            "formName": result["form_name"],
                            "confidence": result["confidence"],
                            "pose": result.get("pose"),
                            "voice": result.get("voice"),
                            "serverTimingMs": result.get("timing_ms"),
                        }
                    )
                )
                continue

            if mtype == "frame":
                if msg.get("sessionId") != session_state.session_id:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": "session_mismatch",
                                "code": WsErrorCode.NO_SESSION,
                            }
                        )
                    )
                    continue

                timestamp_ms = msg.get("timestampMs")
                if not isinstance(timestamp_ms, int):
                    timestamp_ms = None

                b64 = msg.get("imageJpegBase64")
                if not b64 or not isinstance(b64, str):
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": "missing_imageJpegBase64",
                                "code": WsErrorCode.BAD_MESSAGE,
                            }
                        )
                    )
                    continue
                result = await asyncio.to_thread(
                    process_jpeg_frame,
                    session_state,
                    b64,
                    timestamp_ms,
                )

                if result["kind"] == "error":
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": result["message"],
                                "code": result["code"],
                            }
                        )
                    )
                    continue

                if result["kind"] == "no_pose":
                    await websocket.send_json(
                        _j(
                            {
                                "type": "noPose",
                                "sessionId": result["session_id"],
                                "serverTimingMs": result.get("timing_ms"),
                            }
                        )
                    )
                    continue

                await websocket.send_json(
                    _j(
                        {
                            "type": "metrics",
                            "sessionId": result["session_id"],
                            "count": result["count"],
                            "rawCount": result.get("raw_count"),
                            "measurementType": result.get("measurement_type"),
                            "measurementLabel": result.get("measurement_label"),
                            "holdDurationSec": result.get("hold_duration_sec"),
                            "totalHoldTimeSec": result.get("total_hold_time_sec"),
                            "bestHoldSec": result.get("best_hold_sec"),
                            "completedHolds": result.get("completed_holds"),
                            "state": result["state"],
                            "angle": result["angle"],
                            "feedback": result["feedback"],
                            "progress": result["progress"],
                            "formName": result["form_name"],
                            "confidence": result["confidence"],
                            "pose": result.get("pose"),
                            "voice": result.get("voice"),
                            "serverTimingMs": result.get("timing_ms"),
                        }
                    )
                )
                continue

            if mtype == "reset":
                if msg.get("sessionId") != session_state.session_id:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": "session_mismatch",
                                "code": WsErrorCode.NO_SESSION,
                            }
                        )
                    )
                    continue
                session_state.reset_counters()
                await websocket.send_json(
                    _j({"type": "resetAck", "sessionId": session_state.session_id})
                )
                continue

            if mtype == "end":
                if msg.get("sessionId") != session_state.session_id:
                    await websocket.send_json(
                        _j(
                            {
                                "type": "error",
                                "sessionId": session_state.session_id,
                                "message": "session_mismatch",
                                "code": WsErrorCode.NO_SESSION,
                            }
                        )
                    )
                    continue
                summary = session_state.dispatcher.export_session_data()
                wid = UUID(session_state.session_id)
                row = db.get(WorkoutSession, wid)
                if row is not None:
                    finalize_completed_workout(db, row, summary)
                await websocket.send_json(
                    _j(
                        {
                            "type": "ended",
                            "sessionId": session_state.session_id,
                            "summary": summary,
                        }
                    )
                )
                session_state = None
                db_workout = None
                continue

            await websocket.send_json(
                _j(
                    {
                        "type": "error",
                        "sessionId": session_state.session_id,
                        "message": f"unknown_type:{mtype}",
                        "code": WsErrorCode.BAD_MESSAGE,
                    }
                )
            )

    except WebSocketDisconnect:
        pass
    finally:
        abort_active_workout(db, db_workout)
