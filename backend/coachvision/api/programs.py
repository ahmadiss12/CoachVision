"""Program builder (trainer) and Today's-plan (client) endpoints — Phase 2.

The client's daily plan is resolved from their most recent active assignment:
the program cycles from start_date over weeks*7 days, and each prescribed
exercise's targets are adjusted by the readiness engine (rule_v1) so a
fatigued client automatically gets a lighter day.
"""

from datetime import date, datetime, timezone
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession, selectinload

from .deps import get_current_user, require_role
from .schemas import (
    AssignmentResponse,
    AssignProgramRequest,
    PlanExerciseResponse,
    ProgramCreateRequest,
    ProgramResponse,
    TodayPlanResponse,
)
from ..db.models import (
    Exercise,
    Program,
    ProgramAssignment,
    ProgramDay,
    ProgramExercise,
    Session,
    TrainerClient,
    User,
)
from ..db.session import get_db
from ..services.fatigue_engine import predict_rule_v1
from ..services.fatigue_features import (
    fetch_rolling_features,
    last_self_report_from_predictions,
)

router = APIRouter()

# Readiness level -> multiplier applied to prescribed reps/hold seconds.
_ADJUSTMENT_FACTORS = {"low": 1.0, "moderate": 0.85, "high": 0.7}


def _get_active_link(db: DbSession, trainer_id: UUID, client_id: UUID) -> TrainerClient | None:
    return db.scalar(
        select(TrainerClient).where(
            TrainerClient.trainer_id == trainer_id,
            TrainerClient.client_id == client_id,
            TrainerClient.status == "active",
        )
    )


def _load_program(db: DbSession, trainer_id: UUID, program_id: UUID) -> Program:
    program = db.scalar(
        select(Program)
        .options(selectinload(Program.days).selectinload(ProgramDay.exercises))
        .where(Program.id == program_id, Program.trainer_id == trainer_id)
    )
    if not program:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    return program


@router.post("/trainer/programs", response_model=ProgramResponse, status_code=status.HTTP_201_CREATED)
def create_program(
    payload: ProgramCreateRequest,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> ProgramResponse:
    cycle_length = payload.weeks * 7
    seen_days: set[int] = set()
    for day in payload.days:
        if day.day_index > cycle_length:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"dayIndex {day.day_index} is outside the {cycle_length}-day cycle",
            )
        if day.day_index in seen_days:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Duplicate dayIndex {day.day_index}",
            )
        seen_days.add(day.day_index)

    known_exercises = set(db.scalars(select(Exercise.id)).all())
    for day in payload.days:
        for item in day.exercises:
            if item.exercise_id not in known_exercises:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unknown exercise: {item.exercise_id}",
                )

    program = Program(
        trainer_id=trainer.id,
        name=payload.name.strip(),
        description=payload.description,
        weeks=payload.weeks,
    )
    for day in payload.days:
        day_row = ProgramDay(day_index=day.day_index, notes=day.notes)
        for position, item in enumerate(day.exercises):
            day_row.exercises.append(
                ProgramExercise(
                    exercise_id=item.exercise_id,
                    position=position,
                    target_sets=item.target_sets,
                    target_reps=item.target_reps,
                    rest_seconds=item.rest_seconds,
                    external_load_kg=item.external_load_kg,
                    notes=item.notes,
                )
            )
        program.days.append(day_row)

    db.add(program)
    db.commit()
    db.refresh(program)
    return ProgramResponse.model_validate(program)


@router.get("/trainer/programs", response_model=list[ProgramResponse])
def list_programs(
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> list[ProgramResponse]:
    rows = db.scalars(
        select(Program)
        .options(selectinload(Program.days).selectinload(ProgramDay.exercises))
        .where(Program.trainer_id == trainer.id)
        .order_by(Program.created_at.desc())
    ).all()
    return [ProgramResponse.model_validate(row) for row in rows]


@router.delete("/trainer/programs/{program_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_program(
    program_id: UUID,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> None:
    program = _load_program(db, trainer.id, program_id)
    db.delete(program)
    db.commit()


@router.post("/trainer/programs/{program_id}/assign", response_model=AssignmentResponse)
def assign_program(
    program_id: UUID,
    payload: AssignProgramRequest,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> AssignmentResponse:
    program = _load_program(db, trainer.id, program_id)
    if _get_active_link(db, trainer.id, payload.client_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")

    if payload.start_date:
        try:
            start = date.fromisoformat(payload.start_date)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="startDate must be YYYY-MM-DD",
            ) from exc
    else:
        start = datetime.now(timezone.utc).date()

    # One active plan per client (from this trainer): cancel previous ones.
    existing = db.scalars(
        select(ProgramAssignment).where(
            ProgramAssignment.client_id == payload.client_id,
            ProgramAssignment.trainer_id == trainer.id,
            ProgramAssignment.status == "active",
        )
    ).all()
    for row in existing:
        row.status = "cancelled"

    assignment = ProgramAssignment(
        program_id=program.id,
        trainer_id=trainer.id,
        client_id=payload.client_id,
        start_date=start,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return AssignmentResponse(
        id=assignment.id,
        program_id=program.id,
        program_name=program.name,
        client_id=assignment.client_id,
        start_date=assignment.start_date,
        status=assignment.status,
        created_at=assignment.created_at,
    )


@router.get("/trainer/clients/{client_id}/assignments", response_model=list[AssignmentResponse])
def client_assignments(
    client_id: UUID,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> list[AssignmentResponse]:
    if _get_active_link(db, trainer.id, client_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    rows = db.execute(
        select(ProgramAssignment, Program.name)
        .join(Program, Program.id == ProgramAssignment.program_id)
        .where(
            ProgramAssignment.client_id == client_id,
            ProgramAssignment.trainer_id == trainer.id,
        )
        .order_by(ProgramAssignment.created_at.desc())
    ).all()

    counts = dict(
        db.execute(
            select(Session.assignment_id, func.count(Session.id))
            .where(
                Session.user_id == client_id,
                Session.status == "completed",
                Session.assignment_id.is_not(None),
            )
            .group_by(Session.assignment_id)
        ).all()
    )
    today = datetime.now(timezone.utc).date()
    return [
        AssignmentResponse(
            id=row.id,
            program_id=row.program_id,
            program_name=name,
            client_id=row.client_id,
            start_date=row.start_date,
            status=row.status,
            created_at=row.created_at,
            days_elapsed=max(0, (today - row.start_date).days + 1) if row.status == "active" else None,
            sessions_completed=int(counts.get(row.id, 0)),
        )
        for row, name in rows
    ]


@router.delete("/trainer/assignments/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_assignment(
    assignment_id: UUID,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> None:
    assignment = db.get(ProgramAssignment, assignment_id)
    if not assignment or assignment.trainer_id != trainer.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment.status = "cancelled"
    db.commit()


def _client_today(user: User) -> date:
    try:
        tz = ZoneInfo(user.timezone or "UTC")
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).date()


@router.get("/me/plan/today", response_model=TodayPlanResponse)
def today_plan(
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TodayPlanResponse:
    assignment = db.scalar(
        select(ProgramAssignment)
        .where(
            ProgramAssignment.client_id == current_user.id,
            ProgramAssignment.status == "active",
        )
        .order_by(ProgramAssignment.created_at.desc())
        .limit(1)
    )
    if not assignment:
        return TodayPlanResponse(has_plan=False, message="No training plan assigned yet.")

    program = db.scalar(
        select(Program)
        .options(selectinload(Program.days).selectinload(ProgramDay.exercises))
        .where(Program.id == assignment.program_id)
    )
    trainer = db.get(User, assignment.trainer_id)
    trainer_name = trainer.display_name if trainer else None
    if not program:
        return TodayPlanResponse(has_plan=False, message="Assigned program no longer exists.")

    today = _client_today(current_user)
    cycle_length = max(1, program.weeks * 7)
    day_offset = (today - assignment.start_date).days
    if day_offset < 0:
        return TodayPlanResponse(
            has_plan=True,
            rest_day=True,
            program_name=program.name,
            trainer_name=trainer_name,
            assignment_id=assignment.id,
            cycle_length=cycle_length,
            message=f"Your plan starts on {assignment.start_date.isoformat()}.",
        )

    day_index = (day_offset % cycle_length) + 1
    day = next((d for d in program.days if d.day_index == day_index), None)
    if day is None or not day.exercises:
        return TodayPlanResponse(
            has_plan=True,
            rest_day=True,
            program_name=program.name,
            trainer_name=trainer_name,
            assignment_id=assignment.id,
            day_index=day_index,
            cycle_length=cycle_length,
            day_notes=day.notes if day else None,
            message="Rest day — recover well.",
        )

    exercises_by_id = {
        row.id: row
        for row in db.scalars(
            select(Exercise).where(Exercise.id.in_({e.exercise_id for e in day.exercises}))
        ).all()
    }

    plan_items: list[PlanExerciseResponse] = []
    for item in day.exercises:
        rolling = fetch_rolling_features(
            db, current_user.id, item.exercise_id, window_days=14
        )
        self_report = last_self_report_from_predictions(db, current_user.id, item.exercise_id)
        prediction = predict_rule_v1(
            rolling,
            self_report,
            window_days=14,
            exercise_id=item.exercise_id,
        )
        factor = _ADJUSTMENT_FACTORS.get(prediction.fatigue_level, 1.0)
        adjusted_reps = max(1, round(item.target_reps * factor))
        adjusted_sets = item.target_sets if factor >= 0.85 else max(1, item.target_sets - 1)

        note = None
        if factor < 1.0:
            note = (
                f"Reduced from {item.target_sets}×{item.target_reps} because readiness "
                f"is {prediction.fatigue_level} ({prediction.readiness_score}/100)."
            )

        exercise_row = exercises_by_id.get(item.exercise_id)
        plan_items.append(
            PlanExerciseResponse(
                exercise_id=item.exercise_id,
                exercise_name=exercise_row.name if exercise_row else item.exercise_id,
                prescribed_sets=item.target_sets,
                prescribed_reps=item.target_reps,
                adjusted_sets=adjusted_sets,
                adjusted_reps=adjusted_reps,
                rest_seconds=item.rest_seconds,
                external_load_kg=float(item.external_load_kg) if item.external_load_kg is not None else None,
                notes=item.notes,
                readiness_score=prediction.readiness_score,
                fatigue_level=prediction.fatigue_level,
                adjustment_note=note,
            )
        )

    return TodayPlanResponse(
        has_plan=True,
        rest_day=False,
        program_name=program.name,
        trainer_name=trainer_name,
        assignment_id=assignment.id,
        day_index=day_index,
        cycle_length=cycle_length,
        day_notes=day.notes,
        exercises=plan_items,
    )
