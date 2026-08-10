"""WebSocket live workout channel (v1 contract aligned with mobile team)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session as DbSession

from sqlalchemy import select

from coachvision.ai.counters.dispatcher import CONFIG_PRESETS, ExerciseType
from coachvision.benchmark.recorder import ClipRecorder
from coachvision.core.config import settings
from coachvision.core.security import decode_token
from coachvision.db.models import Session as WorkoutSession, TrainerClient, User
from coachvision.db.session import get_db
from coachvision.realtime.connection_manager import live_registry
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
logger = logging.getLogger(__name__)


def _j(payload: dict[str, Any]) -> dict[str, Any]:
    return {"schemaVersion": SCHEMA_VERSION, **payload}


def _new_clip_recorder(session_id: str, exercise_id: str, difficulty: str) -> ClipRecorder | None:
    """A recorder when dataset capture is switched on, otherwise nothing.

    Settings refuses to enable this outside development, so in production this
    is a single boolean check per workout.
    """
    if not settings.clip_recording_enabled:
        return None
    return ClipRecorder(
        session_id=session_id,
        exercise=exercise_id,
        difficulty=difficulty,
        fixtures_dir=Path(settings.clip_recording_dir),
    )


async def _finish_clip_recording(recorder: ClipRecorder | None) -> None:
    """Flush a recorder off the event loop; never let it break the session.

    Only safe on the graceful `end` path. On disconnect the handler task is
    being cancelled and awaiting anything re-raises CancelledError, so use
    :func:`_flush_clip_recording_now` there instead.
    """
    if recorder is None:
        return
    try:
        await asyncio.to_thread(recorder.finish)
    except Exception:  # noqa: BLE001 - a failed recording is not a failed workout
        logger.exception("Clip recording flush failed")


def _flush_clip_recording_now(recorder: ClipRecorder | None) -> None:
    """Write the clip synchronously, for the disconnect path.

    When the client drops, Starlette cancels this handler; the first `await` in
    the teardown then raises CancelledError and the flush is skipped, losing the
    session's frames at random depending on whether a worker thread happened to
    win the race. A blocking write cannot be interrupted that way. It costs a
    single file write on a socket that is already gone, and clip recording only
    runs in development.
    """
    if recorder is None:
        return
    try:
        recorder.finish()
    except Exception:  # noqa: BLE001 - a failed recording is not a failed workout
        logger.exception("Clip recording flush failed")


async def _send_error(
    websocket: WebSocket,
    session_id: str | None,
    message: str,
    code: str,
) -> None:
    await websocket.send_json(
        _j({"type": "error", "sessionId": session_id, "message": message, "code": code})
    )


def _metrics_message(result: dict[str, Any]) -> dict[str, Any]:
    return _j(
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
            "formConfidence": result.get("form_confidence"),
            "formProbabilities": result.get("form_probabilities"),
            "formSource": result.get("form_source"),
            "confidence": result["confidence"],
            "pose": result.get("pose"),
            "voice": result.get("voice"),
            "serverTimingMs": result.get("timing_ms"),
        }
    )


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


def _authenticate(token: str) -> UUID:
    """Decode the access token and return the user id; raises on any problem."""
    payload = decode_token(token)
    if payload.get("token_type") != "access":
        raise ValueError("Invalid token type")
    return UUID(str(payload["sub"]))


@router.websocket("/ws/live")
async def websocket_live(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
    db: DbSession = Depends(get_db),
) -> None:
    await websocket.accept()

    session_state: LiveSessionState | None = None
    db_workout: WorkoutSession | None = None
    clip_recorder: ClipRecorder | None = None

    try:
        try:
            user_uuid = _authenticate(token)
        except Exception as exc:  # noqa: BLE001
            await _send_error(websocket, None, str(exc), WsErrorCode.INVALID_TOKEN)
            await websocket.close(code=4401)
            return

        while True:
            raw = await websocket.receive_text()
            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(
                    websocket,
                    session_state.session_id if session_state else None,
                    "invalid_json",
                    WsErrorCode.BAD_MESSAGE,
                )
                continue

            mtype = msg.get("type")

            if mtype == "start":
                if session_state is not None:
                    await _send_error(
                        websocket,
                        session_state.session_id,
                        "session_already_active",
                        WsErrorCode.ALREADY_ACTIVE,
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
                        await _send_error(
                            websocket, None, "invalid_sessionId", WsErrorCode.BAD_MESSAGE
                        )
                        continue

                try:
                    ExerciseType.from_string(exercise_name)
                except ValueError as exc:
                    await _send_error(
                        websocket, None, str(exc), WsErrorCode.UNSUPPORTED_EXERCISE
                    )
                    continue

                if difficulty not in CONFIG_PRESETS:
                    await _send_error(
                        websocket,
                        None,
                        f"Invalid difficulty: {difficulty}",
                        WsErrorCode.BAD_DIFFICULTY,
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
                    await _send_error(
                        websocket, None, err or "start_failed", err or WsErrorCode.START_FAILED
                    )
                    continue

                db_workout = workout
                try:
                    session_state = LiveSessionState(
                        session_id=str(workout.id),
                        exercise_name=exercise_id,
                        difficulty=difficulty,
                        include_pose=bool(msg.get("includePose", False)),
                    )
                except ValueError as exc:
                    abort_active_workout(db, workout)
                    db_workout = None
                    await _send_error(
                        websocket, str(workout.id), str(exc), WsErrorCode.UNSUPPORTED_EXERCISE
                    )
                    continue
                live_registry.register_session(str(workout.id), str(user_uuid), exercise_id)
                clip_recorder = _new_clip_recorder(str(workout.id), exercise_id, difficulty)
                await websocket.send_json(_j({"type": "started", "sessionId": str(workout.id)}))
                continue

            if session_state is None:
                await _send_error(websocket, None, "send_start_first", WsErrorCode.NO_SESSION)
                continue

            if msg.get("sessionId") != session_state.session_id and mtype in (
                "pose",
                "frame",
                "reset",
                "end",
            ):
                await _send_error(
                    websocket, session_state.session_id, "session_mismatch", WsErrorCode.NO_SESSION
                )
                continue

            if mtype == "pose":
                timestamp_ms = msg.get("timestampMs")
                if not isinstance(timestamp_ms, int):
                    timestamp_ms = None

                client_inference_ms = msg.get("clientInferenceMs")
                if not isinstance(client_inference_ms, (int, float)):
                    client_inference_ms = None

                landmarks = msg.get("landmarks")
                if clip_recorder is not None:
                    # Buffered in memory, written once at the end -- a disk
                    # write per frame would add latency to the live loop.
                    clip_recorder.record(landmarks, timestamp_ms)
                # Offloaded like the JPEG path: the counters and the XGBoost
                # form model are synchronous CPU work, and running them inline
                # stalls the event loop for every other live session. Messages
                # for one socket are still processed strictly in order, so the
                # per-session state cannot interleave with itself.
                result = await asyncio.to_thread(
                    process_pose_landmarks,
                    session_state,
                    landmarks,
                    timestamp_ms,
                    client_inference_ms,
                )

                if result["kind"] == "error":
                    await _send_error(
                        websocket, session_state.session_id, result["message"], result["code"]
                    )
                    continue

                metrics = _metrics_message(result)
                await websocket.send_json(metrics)
                await live_registry.broadcast(session_state.session_id, metrics)
                continue

            if mtype == "frame":
                timestamp_ms = msg.get("timestampMs")
                if not isinstance(timestamp_ms, int):
                    timestamp_ms = None

                b64 = msg.get("imageJpegBase64")
                if not b64 or not isinstance(b64, str):
                    await _send_error(
                        websocket,
                        session_state.session_id,
                        "missing_imageJpegBase64",
                        WsErrorCode.BAD_MESSAGE,
                    )
                    continue
                result = await asyncio.to_thread(
                    process_jpeg_frame,
                    session_state,
                    b64,
                    timestamp_ms,
                )

                if result["kind"] == "error":
                    await _send_error(
                        websocket, session_state.session_id, result["message"], result["code"]
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

                metrics = _metrics_message(result)
                await websocket.send_json(metrics)
                await live_registry.broadcast(session_state.session_id, metrics)
                continue

            if mtype == "reset":
                session_state.reset_counters()
                await websocket.send_json(
                    _j({"type": "resetAck", "sessionId": session_state.session_id})
                )
                continue

            if mtype == "end":
                summary = session_state.dispatcher.export_session_data()
                wid = UUID(session_state.session_id)
                row = db.get(WorkoutSession, wid)
                if row is not None:
                    finalize_completed_workout(db, row, summary)
                ended_message = _j(
                    {
                        "type": "ended",
                        "sessionId": session_state.session_id,
                        "summary": summary,
                    }
                )
                await websocket.send_json(ended_message)
                await live_registry.broadcast(session_state.session_id, ended_message)
                live_registry.unregister_session(session_state.session_id)
                await _finish_clip_recording(clip_recorder)
                clip_recorder = None
                session_state = None
                db_workout = None
                continue

            await _send_error(
                websocket,
                session_state.session_id,
                f"unknown_type:{mtype}",
                WsErrorCode.BAD_MESSAGE,
            )

    except WebSocketDisconnect:
        pass
    finally:
        if session_state is not None:
            # Client disconnected mid-workout: tell observers and clean up.
            await live_registry.broadcast(
                session_state.session_id,
                _j({"type": "ended", "sessionId": session_state.session_id, "summary": None}),
            )
            live_registry.unregister_session(session_state.session_id)
        # An abandoned workout still recorded real reps, and a clip is scored
        # against what a human labels rather than against a completed session,
        # so the frames are worth keeping. Written synchronously: this runs
        # while the task is being cancelled, where an await would be skipped.
        _flush_clip_recording_now(clip_recorder)
        abort_active_workout(db, db_workout)


@router.websocket("/ws/observe/{session_id}")
async def websocket_observe(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(..., description="JWT access token of a trainer or admin"),
    db: DbSession = Depends(get_db),
) -> None:
    """Read-only mirror of a client's live workout for their trainer.

    Access rules: the caller must hold a valid access token for a trainer
    (with an active link to the session's owner) or an admin, and the
    session must currently be live. Observers only receive; any message
    they send is ignored (used as keep-alive).
    """
    await websocket.accept()

    try:
        payload = decode_token(token)
        if payload.get("token_type") != "access":
            raise ValueError("Invalid token type")
        viewer_id = UUID(str(payload["sub"]))
    except Exception:  # noqa: BLE001
        await _send_error(websocket, session_id, "Invalid token", WsErrorCode.INVALID_TOKEN)
        await websocket.close(code=4401)
        return

    viewer = db.get(User, viewer_id)
    if viewer is None or viewer.role not in ("trainer", "admin"):
        await _send_error(websocket, session_id, "Trainer or admin required", WsErrorCode.INVALID_TOKEN)
        await websocket.close(code=4403)
        return

    info = live_registry.get_session(session_id)
    if info is None:
        await _send_error(websocket, session_id, "Session is not live", "NOT_LIVE")
        await websocket.close(code=4404)
        return

    if viewer.role == "trainer":
        link = db.scalar(
            select(TrainerClient).where(
                TrainerClient.trainer_id == viewer.id,
                TrainerClient.client_id == UUID(info.user_id),
                TrainerClient.status == "active",
            )
        )
        if link is None:
            await _send_error(websocket, session_id, "Session is not live", "NOT_LIVE")
            await websocket.close(code=4404)
            return

    live_registry.add_observer(session_id, websocket)
    await websocket.send_json(
        _j(
            {
                "type": "observing",
                "sessionId": session_id,
                "exerciseId": info.exercise_id,
                "startedAt": info.started_at,
            }
        )
    )
    try:
        while True:
            # Observers are receive-only; incoming frames are keep-alives.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        live_registry.remove_observer(session_id, websocket)
