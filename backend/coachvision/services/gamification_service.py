"""Gamification service for streak/XP/badges (rule-based v1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from ..db.models import Achievement, Session as WorkoutSession, Streak, UserAchievement, XpEvent


@dataclass
class GamificationService:
    """Updates streaks, XP, and achievements."""

    db: DbSession

    def _ensure_achievement_catalog(self) -> None:
        """Seed a minimal set of achievements if missing."""
        seeds = [
            (
                "first_workout",
                "First Workout",
                "Complete your first coached session.",
                50,
            ),
            (
                "streak_3",
                "3-Day Streak",
                "Train on 3 different days in a row.",
                80,
            ),
            (
                "streak_7",
                "7-Day Streak",
                "Train on 7 different days in a row.",
                200,
            ),
            (
                "volume_200_reps",
                "200 Reps (Week)",
                "Accumulate 200 reps within 7 days on a single exercise.",
                120,
            ),
        ]
        existing = set(self.db.scalars(select(Achievement.id)).all())
        for aid, name, desc, xp in seeds:
            if aid in existing:
                continue
            self.db.add(Achievement(id=aid, name=name, description=desc, xp_reward=xp))
        self.db.flush()

    def _award(self, user_id: UUID, achievement_id: str, *, session_id: UUID | None) -> None:
        already = self.db.scalars(
            select(UserAchievement)
            .where(
                UserAchievement.user_id == user_id,
                UserAchievement.achievement_id == achievement_id,
            )
            .limit(1)
        ).first()
        if already:
            return
        self.db.add(UserAchievement(user_id=user_id, achievement_id=achievement_id))

        ach = self.db.get(Achievement, achievement_id)
        if ach and ach.xp_reward:
            self.db.add(
                XpEvent(
                    user_id=user_id,
                    session_id=session_id,
                    source=f"achievement:{achievement_id}",
                    points=int(ach.xp_reward),
                    meta={"achievementId": achievement_id},
                )
            )

    def _update_streak(self, user_id: UUID, activity_date) -> Streak:
        row = self.db.get(Streak, user_id)
        now = datetime.now(timezone.utc)
        if row is None:
            row = Streak(
                user_id=user_id,
                current_streak_days=1,
                longest_streak_days=1,
                last_activity_date=activity_date,
                updated_at=now,
            )
            self.db.add(row)
            return row

        last = row.last_activity_date
        if last == activity_date:
            row.updated_at = now
            self.db.add(row)
            return row

        if last is None:
            row.current_streak_days = 1
        else:
            delta_days = (activity_date - last).days
            if delta_days == 1:
                row.current_streak_days += 1
            else:
                row.current_streak_days = 1

        row.longest_streak_days = max(row.longest_streak_days, row.current_streak_days)
        row.last_activity_date = activity_date
        row.updated_at = now
        self.db.add(row)
        return row

    def _award_streak_achievements(self, user_id: UUID, streak: Streak, *, session_id: UUID) -> None:
        if streak.current_streak_days >= 3:
            self._award(user_id, "streak_3", session_id=session_id)
        if streak.current_streak_days >= 7:
            self._award(user_id, "streak_7", session_id=session_id)

    def _award_volume_achievements(self, workout: WorkoutSession) -> None:
        # Weekly volume for this exercise
        from datetime import timedelta

        cutoff = workout.ended_at - timedelta(days=7)
        reps_7d = (
            self.db.query(func.coalesce(func.sum(WorkoutSession.total_reps), 0))
            .filter(
                WorkoutSession.user_id == workout.user_id,
                WorkoutSession.exercise_id == workout.exercise_id,
                WorkoutSession.status == "completed",
                WorkoutSession.ended_at.is_not(None),
                WorkoutSession.ended_at >= cutoff,
            )
            .scalar()
        )
        try:
            reps_7d_i = int(reps_7d or 0)
        except Exception:  # noqa: BLE001
            reps_7d_i = 0
        if reps_7d_i >= 200:
            self._award(workout.user_id, "volume_200_reps", session_id=workout.id)

    def _grant_session_xp(self, workout: WorkoutSession) -> None:
        # Avoid double-grant: one 'session_completed' per session.
        existing = self.db.scalars(
            select(XpEvent)
            .where(
                XpEvent.user_id == workout.user_id,
                XpEvent.session_id == workout.id,
                XpEvent.source == "session_completed",
            )
            .limit(1)
        ).first()
        if existing:
            return

        reps = int(workout.total_reps or 0)
        base = 20
        rep_xp = min(200, reps)  # 1 xp per rep capped
        form_bonus = 0
        if workout.avg_form_score is not None and float(workout.avg_form_score) >= 85:
            form_bonus = 25
        total = int(base + rep_xp + form_bonus)

        self.db.add(
            XpEvent(
                user_id=workout.user_id,
                session_id=workout.id,
                source="session_completed",
                points=total,
                meta={
                    "base": base,
                    "repXp": rep_xp,
                    "formBonus": form_bonus,
                    "exerciseId": workout.exercise_id,
                },
            )
        )

    def apply_on_completed_session(self, workout_id: UUID) -> None:
        w = self.db.get(WorkoutSession, workout_id)
        if w is None or w.status != "completed":
            return
        if w.ended_at is None:
            return

        self._ensure_achievement_catalog()

        # streak: uses UTC calendar day of session end
        streak = self._update_streak(w.user_id, w.ended_at.date())

        # xp for this session
        self._grant_session_xp(w)

        # achievements
        self._award(w.user_id, "first_workout", session_id=w.id)
        self._award_streak_achievements(w.user_id, streak, session_id=w.id)

        # optional volume achievement (safe to skip if date funcs differ)
        try:
            self._award_volume_achievements(w)
        except Exception:  # noqa: BLE001
            # keep gamification robust even if DB date functions differ
            pass

        self.db.flush()

