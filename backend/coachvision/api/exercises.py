"""Exercise catalog endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .schemas import ExerciseResponse
from ..db.models import Exercise
from ..db.session import get_db

router = APIRouter(prefix="/exercises")


@router.get("", response_model=list[ExerciseResponse])
def list_exercises(db: Session = Depends(get_db)) -> list[ExerciseResponse]:
    exercises = db.scalars(select(Exercise).order_by(Exercise.id.asc())).all()
    return [ExerciseResponse.model_validate(ex) for ex in exercises]


@router.get("/{exercise_id}", response_model=ExerciseResponse)
def get_exercise(exercise_id: str, db: Session = Depends(get_db)) -> ExerciseResponse:
    exercise = db.get(Exercise, exercise_id)
    if not exercise:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exercise not found")
    return ExerciseResponse.model_validate(exercise)

