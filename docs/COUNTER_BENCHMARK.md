# Counter Accuracy Benchmark

The twelve exercise counters are hand-tuned angle state machines. Before this
harness existed there was no number anywhere in the repo saying how accurate
they are, which meant no claim about them could be checked and no threshold
change could be shown to be an improvement.

This benchmark replays recorded pose clips through the live counting path and
compares what the counter said against what a human says actually happened.

## Quick start

```powershell
cd backend
python scripts/make_synthetic_clips.py     # generated demo clips, no camera needed
python scripts/benchmark_counters.py --per-clip
```

## The full loop

```text
1. record     enable CLIP_RECORDING_ENABLED, work out in the app
2. label      python scripts/label_clips.py
3. measure    python scripts/benchmark_counters.py --per-clip
4. baseline   python scripts/benchmark_counters.py --out BASELINE.md
5. tune       change thresholds, re-run step 3, compare
```

Step 3 is where a disagreement shows up. `expected` is what you said you did,
`counted` is what the app's counter produced:

```text
clip                                expected  counted   error
-------------------------------------------------------------
rec_squat_7207a420                         7        8      +1  <-- MISS
rec_squat_a91c04f2                        12       12      +0
rec_pushup_5b30e1aa                       10        8      -2  <-- MISS
```

A positive error means the counter over-counted (it saw reps you did not do);
negative means it under-counted (it missed reps you did).

## How it works

`ExerciseCounter.update()` takes `{joint: (x, y)}` plus a confidence float and
returns `(count, state, angle)`. It is a pure function of a landmark sequence,
so no camera and no MediaPipe are needed to exercise it — and the mobile app
already sends exactly those landmarks over the WebSocket. A clip is that stream
saved to disk.

```text
clip.jsonl ──> IMPORTANT_LANDMARKS ──> LandmarkFilter ──> ExerciseDispatcher ──> count
               (pipeline.py)           (One Euro)         (counter FSM)
                                            ^                    ^
                                            └──── ReplayClock ───┘
```

Two things make the number trustworthy:

**It replays the path production uses.** The live pipeline smooths every
landmark with a One Euro filter *before* the counter sees it. The harness
imports `IMPORTANT_LANDMARKS`, `LANDMARK_FILTER_MIN_CUTOFF` and
`LANDMARK_FILTER_BETA` from `realtime/pipeline.py` rather than copying them, so
retuning the pipeline automatically retunes the benchmark. Benchmarking a bare
counter would measure something no user ever experiences.

**It replays on clip time, not wall-clock time.** Every counter takes an
injected `clock` (defaulting to `time.time`), and the harness drives it from
the recorded frame timestamps. A 60-second clip therefore scores identically
whether it replays in 60 seconds or 60 milliseconds.

## Clip format

Two files per clip, under `backend/fixtures/<exercise>/`:

`<clip_id>.jsonl` — one frame per line:

```json
{"t": 0.067, "confidence": 0.94, "landmarks": [[0.51, 0.72, 1.0], ...]}
```

`t` is seconds since clip start; `landmarks` is the same `[x, y, presence]`
array the WebSocket `pose` message already carries, so a recorder is a straight
dump of the client payload.

`<clip_id>.meta.json` — the ground truth:

```json
{
  "exercise": "squat",
  "level": "intermediate",
  "true_reps": 5,
  "camera_angle": "side"
}
```

Hold exercises (plank, wall sit) use `true_hold_sec` instead of `true_reps`.

Coordinates are normalized floats, never pixels, so clips contain no image data
and nothing personally identifiable. That is why they can be committed.

## Metrics

| Metric | Meaning |
| --- | --- |
| Exact | Clips counted exactly right. Holds allow ±2s; reps must match. |
| MAE | Mean absolute error. Errors do not cancel: +2 and −2 is 2.0, not 0. |
| Over | Clips that counted too high — the FSM is retriggering on jitter or rewarding partial reps. |
| Under | Clips that counted too low — a threshold is stricter than the range of motion people actually use. |

Over and under are reported separately because they are different defects that
need opposite fixes. A single mean error hides which one you have.

## Recording real clips

The generated clips validate the harness, **not** the counters. A synthetic
skeleton performs a textbook rep every time — no occlusion, no camera roll, no
half-reps, no bouncing at the bottom. Scoring 100% on them says the replay path
works, nothing more.

### Turning the recorder on

Recording saves the pose landmarks of live workouts straight into the clip
format. In `backend/.env`:

```ini
ENVIRONMENT=development
CLIP_RECORDING_ENABLED=true
CLIP_RECORDING_DIR=fixtures
```

Then work out in the app as normal. Every live session writes
`fixtures/<exercise>/rec_<exercise>_<session>.jsonl` plus a metadata file.

This records body movement data from whoever is using the app, so it is treated
as a data-collection feature: off by default, and the backend **refuses to
start** with it enabled unless `ENVIRONMENT=development`. Turning it on is a
decision, never a default. Landmarks are coordinates, not video, but they are
still a recording of a person.

Frames are buffered in memory and written once when the session ends, so the
live loop does no disk I/O. Sessions abandoned without an `end` message are
still saved — those reps happened.

### Labelling what you recorded

A recorded clip lands with `needs_label: true` and no ground truth. Until it is
labelled the benchmark skips it and lists it as pending rather than scoring it —
an unlabelled clip scored against an empty label would invent an accuracy
number.

```powershell
python scripts/label_clips.py --status   # how many clips, how many labelled
python scripts/label_clips.py            # walk through the unlabelled ones
```

The tool plots the joint angle over time and asks what actually happened. One
dip is one repetition:

```text
Clip 3 of 12: rec_squat_7207a420
  Exercise : squat (reps)
  Recorded : 20.7s, 312 frames, 15.0 fps

  Joint angle over time (one dip = one repetition):
    172 |####                                      ##      ##              ##
        |    #     ###     # #     ###     ##      ##      ##     ###     #
        |     #   ##  #   #  ##   #  #    #  #    #  #    #  #   ##  #   #
        |      # ##    # #     # #    ## #    ## #    # ##    # ##    # #
        |      ###     ###     ###     ###     ###    ###     ###     ###
     86 +--------------------------------------------------------------------

  How many reps did you actually do? (s=skip, q=quit)
  > 8
  Camera angle? [side/front/45/other] (default side): side
  Notes (optional):
  Saved: 8 reps, side view.
```

Hold exercises get the same plot, where each flat plateau is one hold, and are
asked for seconds instead of reps.

The tool never displays what the counter said, and the recorder never writes the
app's own count into the metadata. Both would mean labelling the data with the
very thing the benchmark measures: every clip would score 100% and the number
would be worthless. The plot is raw joint geometry, not the counter's decision
about it.

`--relabel` revisits clips that already have a label. Editing the `.meta.json`
by hand does the same job if you prefer.

### What to capture

Target **15–25 clips per exercise** and vary deliberately:

- camera angle: front, side, 45°
- distance: close and far
- phone height: hip and chest
- lighting: bright, dim, backlit
- tempo: fast and slow
- form: clean reps, and deliberately sloppy ones

Label from the recording, not from what the app displayed during capture —
otherwise the label comes from the thing being measured.

If you cannot reach ~15 clips for all twelve exercises, cover four exercises
properly rather than twelve badly. Five clips is noise, not a measurement.

Recorded clips under `backend/fixtures/` are **not** git-ignored — the
labelling is the work, and the files hold coordinates rather than image data.
Only the generated `synthetic_*` clips are ignored, since one command
reproduces them.

## Regression gate

Once real clips exist, record a baseline and wire it into CI:

```powershell
python scripts/benchmark_counters.py --out BASELINE.md
python scripts/benchmark_counters.py --check BASELINE.md   # exits 1 on regression
```

The gate compares per-exercise exact-rates and fails the build if any exercise
scores below its baseline. A baseline records reality, not perfection — a known
defect can sit in the baseline while the gate still prevents new ones.

## Findings

### Hold time is frame-counted at an assumed 30 fps

`plank.py:329` and `wall_sit.py:306` both accumulate hold time as a hard-coded
`1.0 / 30` per frame rather than from elapsed time:

```python
self._total_hold_time += (1.0 / 30)  # Approximate per frame
```

The live rate is roughly 15 Hz (see the One Euro tuning note in
`realtime/pipeline.py`), so `total_hold_time` runs at about half speed.

A single unbroken hold hides this, because `export_session_data()` takes the
`max` of `total_hold_time` and the clock-derived `best_hold`. It surfaces as
soon as the hold breaks — which is the normal case for a plank:

```text
fixtures/plank/synthetic_plank_60sec_3holds  (three 20s holds at 15 fps)
  total_hold_time (frame-counted) : 29.8s
  best_hold       (clock-based)   : 19.8s
  persisted total_seconds         : 30s     <-- should be 60s
```

The user is credited with half the plank they actually did, and the wrong value
reaches persisted session data via `ws_persistence`. The fix is to accumulate
`current_time - last_frame_time` from the injected clock, which is now
available. Left unfixed here because it changes user-visible history values and
warrants its own change; the fixture documents it in the meantime.

### `use_both_legs` uses only the left leg

`squat.py:228` reads:

```python
if self.config.use_both_legs:
    # Average both sides: robust to single-side errors
    knee_angle = left_angle
```

The comment says average; the code discards `right_angle` entirely. Anyone
filmed from their right side is measured on the occluded leg. Worth a
front/side/45° clip set to quantify before changing, which is exactly what this
harness is for.
