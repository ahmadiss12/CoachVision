# Google Play store listing — CoachVision

Copy each block into Play Console. Character limits are enforced by Google.

---

## App name (max 30)

```
CoachVision
```

Alternative if you want keywords in the name (Play allows it, but keep it clean):

```
CoachVision: AI Form Coach
```

---

## Short description (max 80)

```
AI form coaching that counts your reps and checks your technique in real time.
```

Alternatives:

```
Count reps and check your workout form in real time, using just your camera.
```

```
Your camera counts the reps. The AI watches your form. No equipment needed.
```

---

## Full description (max 4000)

```
CoachVision turns your phone camera into a form coach. Set it down, start your
set, and the app counts every rep while checking how you actually move.

No wearables. No extra equipment. Just your phone.


WHAT IT DOES

Live rep counting
Your reps are counted as you go, with the current phase of the movement and your
joint angle shown on screen. A skeleton overlay tracks your body in real time so
you can see exactly what the app sees.

Real-time form feedback
CoachVision checks your technique while you train and speaks up when something
slips, with cues like "go deeper" or "keep your chest up". Squats are analysed
by a trained machine-learning model that recognises specific faults, including
shallow depth, forward lean, knees caving inward, heels lifting, and uneven
left-right loading.

Post-workout review
After every session you get a plain-language summary: what you did well, the
issues that came up most often, and a short list of cues to focus on next time.

Recovery and readiness
Log your sleep, soreness, and stress, and CoachVision combines them with your
recent training load to estimate how ready you are. It suggests whether to push,
hold steady, or take an easier session, and explains the reasoning behind the
number rather than hiding it.

Progress tracking
Full workout history, body weight and body-fat logging, BMI, and goal tracking.
Export a daily coaching report as a PDF whenever you want a record.


EXERCISES SUPPORTED

Rep-based: squat, push-up, lunge, deadlift, bicep curl, shoulder press, sit-up,
jumping jack, high knees, mountain climber.

Hold-based: plank, wall sit, timed with a live counter.

Each exercise has beginner, intermediate, and advanced settings, so the depth
and range expected of you changes with your level.


YOUR CAMERA STAYS PRIVATE

This matters, so we will be direct about it: video from your workout is never
uploaded and never stored. Pose detection runs entirely on your device. Only
numeric joint coordinates leave your phone so the app can count reps and score
form. There is no advertising, no third-party tracking, and no analytics SDK.

You can permanently delete your account and all of your data at any time from
Settings, without emailing anyone or waiting for approval.


WORKING WITH A TRAINER

If you train with a coach, they can build programs, assign them to your week,
review your completed sessions, and message you in the app. Trainers see only
the clients who have accepted their invitation, and you can end the link at any
time.


GOOD TO KNOW

- Works best with your whole body in frame and reasonable lighting.
- The camera is used only while a workout screen is open.
- An internet connection is required during workouts.
- CoachVision offers general fitness feedback. It is not a medical device and
  does not diagnose, treat, or prevent any condition. Talk to a qualified
  professional before starting a new exercise programme, especially if you have
  an injury or a health condition.
```

---

## Play Console — Data Safety answers

Based on what the code actually collects.

**Data collected and sent off the device**

| Category | Type | Collected | Shared | Purpose | Optional |
|---|---|---|---|---|---|
| Personal info | Email address | Yes | No | Account management | Required |
| Personal info | Name (display name) | Yes | No | Account management, app functionality | Required |
| Personal info | Photos (avatar) | Yes | No | App functionality | Optional |
| Health & fitness | Fitness info | Yes | No | App functionality | Required |
| Messages | Other in-app messages | Yes | No | App functionality | Optional |
| App activity | Other user-generated content | Yes | No | App functionality | Required |

**Answer these as follows**

- Is all data encrypted in transit? **Yes** (HTTPS / WSS)
- Do you provide a way for users to request data deletion? **Yes** — supply the
  `/delete-account` URL
- Camera: the app requests camera permission, but video is processed on-device
  and never transmitted, so it is **not** declared as collected data. Only the
  derived fitness metrics are declared, under Health & fitness.

**Content rating** — answer the questionnaire honestly; a fitness app with no
violent, sexual, or gambling content normally lands at Everyone / PEGI 3.

**Ads** — declare that the app contains **no ads**.

**Target audience** — 13+ or 18+. Must match the children's clause in the
privacy policy, which currently states the app is not directed at under-13s.

---

## Required URLs

| Field | Value |
|---|---|
| Privacy policy | `https://web-ahmadiss12s-projects.vercel.app/privacy` |
| Account deletion | `https://web-ahmadiss12s-projects.vercel.app/delete-account` |
| Support email | ahmaisma555@gmail.com |

Verified live and publicly reachable (HTTP 200, no login redirect).

Use the domain above, **not** the per-deployment URL that `vercel --prod`
prints. Deployment URLs look like `web-ovj85f24m-...vercel.app` and change on
every deploy, so a listing pointing at one would break the next time the site
is deployed. `web-ahmadiss12s-projects.vercel.app` always tracks the current
production deployment.

---

## Graphics checklist

| Asset | Spec | Status |
|---|---|---|
| App icon | 512x512 PNG | Ready — `play-store-icon-512.png` |
| Feature graphic | 1024x500 PNG/JPEG | Ready — `play-feature-graphic-1024x500.png` |
| Phone screenshots | 2-8, min 320px per side, max 2:1 ratio | **Needs recapture at device resolution** |
