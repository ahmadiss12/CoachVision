# Exercises And Fatigue Logic

CoachVision treats exercises differently based on what the user is actually doing.

## Exercise Matrix

| Exercise | Live metric | External load input |
| --- | --- | --- |
| Squat | Reps | Yes |
| Lunge | Reps | Yes |
| Deadlift | Reps | Yes |
| Bicep curl | Reps | Yes |
| Shoulder press | Reps | Yes |
| Push-up | Reps | No |
| Sit-up | Reps | No |
| Jumping jack | Reps | No |
| High knees | Reps | No |
| Mountain climber | Reps | No |
| Plank | Hold seconds | No |
| Wall sit | Hold seconds | No |

## Hold Exercises

Plank and wall sit do not count reps. When the form is correct, the counter starts a timer. If form breaks, the hold stops. The live screen shows `SEC`, and the session summary stores the tracked hold seconds as the primary output.

The backend WebSocket metrics include:

- `measurementType`: `hold`
- `measurementLabel`: `SEC`
- `count`: current/best hold seconds for the UI
- `rawCount`: completed hold sessions
- `holdDurationSec`
- `totalHoldTimeSec`
- `bestHoldSec`
- `completedHolds`

## Weighted Exercises

Only load-bearing exercises accept external weight. The mobile setup screen hides the load picker for non-load exercises, and the backend also ignores `externalLoadKg` unless the exercise supports load.

The fatigue rule uses:

```text
external load ratio = externalLoadKg / bodyWeightKg
```

If body weight is available, this ratio creates a readiness penalty. If body weight is not available, the engine estimates the penalty from the absolute load. This means a 20 kg dumbbell squat should reduce the next target more than a bodyweight squat, while push-ups and planks are not accidentally penalized by a stale load value.

## Next-Session Target

The mobile fatigue plan converts readiness into a target:

- High fatigue: reduce output to a recovery target.
- Moderate fatigue: use a conservative target.
- Low fatigue: try to match or slightly beat the last session.

For hold exercises, the target unit is seconds. For dynamic exercises, the target unit is reps.
