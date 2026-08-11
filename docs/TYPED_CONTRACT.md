# Typed Client Contract

The backend knows exactly what its payloads look like — Pydantic schemas, a
generated OpenAPI document at `/v1/openapi.json`, and `WS_CONTRACT.md` for the
realtime channel. Until this work, nothing enforced any of that on the side
that consumes it: both clients were plain JavaScript.

The cost of that gap is already in the history. Commit `32f0cf9`:

> Fix live view crash: render voice cue text, not the `{label, text}` object

The server sends `voice` as an object. The live screen rendered it directly,
React cannot render an object, and the workout screen crashed. That is a
compile error now, not a debugging session.

## Status

| Area | State |
| --- | --- |
| Mobile: TypeScript configured | Done |
| Mobile: WebSocket message types | Done |
| Mobile: live WS client migrated | Done |
| Mobile: REST client generated from OpenAPI | Not started |
| Mobile: screens migrated | Not started |
| Web dashboard | Not started |

The app is still mostly JavaScript. `allowJs` is on and `checkJs` is off, so
`.js` and `.ts` sit side by side and files convert one at a time instead of in
one rewrite.

## Running it

```powershell
cd mobile
npm run typecheck
```

CI runs this on every change under `mobile/` (`.github/workflows/mobile-ci.yml`).

## The WebSocket types

OpenAPI does not describe WebSocket frames, so `src/services/ws/messages.ts` is
maintained by hand and must be updated alongside `backend/coachvision/api/
ws_live.py`. Two things about that file are load-bearing.

### Nullable, not optional

The server builds every metrics key explicitly in `_metrics_message`, and Python
`None` serialises to JSON `null`. So the key is *always present* and the value
may be `null`:

```python
"holdDurationSec": result.get("hold_duration_sec"),   # -> null, not absent
```

```ts
holdDurationSec: number | null;   // correct
holdDurationSec?: number;         // wrong: the key is never missing
```

The difference matters: with `?:`, `payload.holdDurationSec` reads as `number`
when the value is really `null`, and `'holdDurationSec' in payload` looks like a
meaningful check when it is always true.

### Reps and holds are different messages

`measurementType` discriminates them. A squat carries `null` in every hold
field; a plank carries numbers. Modelling that as a union means reading
`bestHoldSec` off a rep message does not compile:

```ts
export type MetricsMessage = RepMetricsMessage | HoldMetricsMessage;

function holdSeconds(m: MetricsMessage): number {
  if (isHoldMetrics(m)) {
    return m.totalHoldTimeSec;   // narrowed, readable
  }
  return 0;
}
```

This encodes the reps-versus-seconds split that already runs through the whole
app — the live screen, the summary, the fatigue target — into the type system
rather than leaving it to prose in `EXERCISES_AND_FATIGUE.md`.

## The type tests

`src/services/ws/messages.type-test.ts` contains no runtime code. It uses
`@ts-expect-error`, which **fails the build when the line below it stops being
an error**, so each directive pins a mistake the contract must keep rejecting:

```ts
// @ts-expect-error voice is a VoiceCue object, never a string.
export const voiceAsString: string = metrics.voice;
```

Weaken `voice` to `any` and `tsc` reports `Unused '@ts-expect-error' directive`.
The tests fail when the types get *looser*, which is the direction this kind of
file usually rots in. `npm run typecheck` is the whole test run; there is no
extra tooling.

## Adding to it

When a WebSocket message changes on the server:

1. Update `messages.ts` to match `ws_live.py`.
2. Add a `@ts-expect-error` case to `messages.type-test.ts` for whatever the new
   shape must reject.
3. Run `npm run typecheck` and fix the call sites it flags.

Keep `@ts-expect-error` cases on a single line where the error lands on an inner
property — the directive only covers the line immediately after it.

## What is left

**REST client from OpenAPI.** The spec already exists, so this is mostly
configuration:

```powershell
npx openapi-typescript http://127.0.0.1:8001/v1/openapi.json -o src/services/api/schema.d.ts
```

Then `src/services/api/*.js` becomes typed against the real backend, and a CI
step that regenerates and diffs would catch a backend change that breaks the app
in the pull request rather than in production.

**Screens.** Convert as they are touched, service layer first — that is where
wire data enters and where a wrong assumption does the most damage.

**Web dashboard.** Same approach; it shares the REST surface and the live
observer socket.
