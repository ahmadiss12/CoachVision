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
| Mobile: REST types generated from OpenAPI | Done |
| Mobile: all 11 API service files migrated | Done |
| Mobile: `WorkoutLiveScreen` migrated | Done |
| Mobile: remaining screens | Not started |
| Web dashboard | Not started |

Every module that touches the network is typed. `config.js` stays JavaScript —
it reads environment variables and builds URLs, and never handles wire data.

The app is still mostly JavaScript. `allowJs` is on and `checkJs` is off, so
`.js` and `.ts` sit side by side and files convert one at a time instead of in
one rewrite.

## Running it

```powershell
cd mobile
npm run typecheck      # check every call site
npm test               # unit tests
npm run generate:api   # after a backend schema change
```

```powershell
cd backend
python scripts/export_openapi.py    # after changing any Pydantic model
```

## The REST types

`src/services/api/schema.d.ts` is generated, never edited. It comes from
`backend/openapi.json`, which is exported straight out of the FastAPI app — so
the client's types are the server's own Pydantic models rather than a hand-kept
copy that drifts.

The spec is exported offline rather than fetched from a running server, so CI
can regenerate and diff without booting the app, and a schema change shows up as
a reviewable diff in the pull request that causes it.

### Two gates, one guarantee

Neither check is useful alone; together they mean the client's types match the
running server.

| Check | Where | Catches |
| --- | --- | --- |
| `export_openapi.py --check` | backend CI | `openapi.json` drifting from the Pydantic models |
| regenerate + `git diff --exit-code` | mobile CI | `schema.d.ts` drifting from `openapi.json` |
| `npm run typecheck` | mobile CI | call sites that no longer match the schema |

Renaming a field on the server now fails the pull request:

```text
1. backend CI : openapi.json is out of date with the app.
2. after regenerating:
   sessions.ts(53,25): Property 'exerciseId' does not exist ... Did you mean 'exercise'?
```

### The API is not consistently cased

Generating the types surfaced this: of 263 schema properties, **245 are
camelCase and 18 are snake_case**. The snake_case ones are concentrated in auth
and user profile payloads:

```text
TokenPair.access_token, TokenPair.refresh_token, TokenPair.token_type
RegisterRequest.display_name, RefreshRequest.refresh_token
UserMeResponse.*, UpdateUserMeRequest.*
ExerciseResponse.default_difficulty
```

That inconsistency is why the client used to read
`payload.exerciseId ?? payload.exercise_id` on every field — nobody was certain
which the server sent, so it hedged everywhere. Checking each schema showed
where the hedging was real and where it was dead code:

| Module | Wire casing | Dead fallbacks removed |
| --- | --- | --- |
| `sessions.ts` | camelCase | ~14 |
| `reports.ts` | camelCase | ~40 |
| `fatigue.ts` | camelCase | ~8 |
| `auth.ts` | **snake_case** | 3 (the camelCase side was dead) |
| `exercises.ts` | **snake_case** (`default_difficulty`) | 1 |
| `users.ts` | **snake_case** | — |

So the hedging was backwards in places: `auth.ts` and `exercises.ts` were
falling back to camelCase keys that never arrive, while `reports.ts` was
guarding against snake_case that never arrives either. Every branch was dead in
one direction or the other.

Worth normalising the API on one convention eventually. The generated types make
that a mechanical change now, since every affected call site fails to compile.

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

## Converting a screen: WorkoutLiveScreen

The screen was 1,370 lines with two untyped boundaries in it — the WebSocket and
the WebView bridge — and no tests. Renaming it alone would have bought little,
since the interesting logic was tangled into the component. It was split first:

| File | What it holds |
| --- | --- |
| `workout-live/pose-webview-html.ts` | The 400-line WebView page, moved verbatim |
| `workout-live/presentation.ts` | Tone, colour and status-label rules |
| `workout-live/webview-bridge.ts` | WebView message types and a validating parser |
| `state/live-metrics.ts` | The wire → app-state mapping and the render throttle |
| `WorkoutLiveScreen.tsx` | The component |

Everything except the component is pure, so it is unit tested — 50 tests, the
mobile app's first.

### What the tests found

**`Number(null)` is `0`.** This bit the same code twice:

```js
formConfidence: Number.isFinite(Number(payload.formConfidence)) ? Number(...) : null
```

`formConfidence: null` means the XGBoost model did not run. `Number(null)` is
`0` and `Number.isFinite(0)` is true, so "no model" was recorded as "0%
confidence". Latent — nothing renders it yet — but `app-state.jsx` defaults it
to `null`, so the code plainly means the two to differ.

The same trap then appeared in the *replacement*: a `finiteOr` helper turned a
missing `angle` into 0 degrees instead of falling back to 180. That one was
caught by the parity test, not by review.

**NaN reached the UI.** `Number(payload.angle ?? 180)` passes NaN straight
through, because `??` only catches null and undefined. A NaN angle rendered as
"NaN deg". The replacement rejects non-finite values.

### The parity test

`state/__tests__/live-metrics-parity.test.ts` keeps a copy of the original
inline mapping and asserts the new one produces identical output across
representative frames. Differences have to be deliberate and named: the file
documents two, and pins both directions of each.

That is what makes a conversion like this checkable rather than hopeful.

### One behaviour change

A `pose` message whose `landmarks` is missing or not an array is now rejected at
the boundary, so it no longer marks the WebView as ready. Previously it set
`ready` and then dropped the frame further down. Rejecting it is the safer
reading: a page posting malformed poses is not working, and the frame must not
reach the server.

## Converting a file

The pattern, using `sessions.ts` as the reference:

1. Rename `.js` to `.ts`. Imports are extension-less, so callers do not change.
2. Name the response type from the generated schema:
   `type SessionResponse = Schemas['SessionResponse']`.
3. Pass it to the request: `apiRequest<SessionResponse>('/sessions')`.
4. Run `npm run typecheck` and fix what it flags.

Step 4 is where the value is. Converting `client.js` surfaced a real bug: it
passed `body: undefined` to `fetch` on every GET. Setting a key to `undefined`
is not the same as omitting it, and `RequestInit.body` does not accept it —
the request init is now built conditionally.

## What is left

**Screens.** `WorkoutLiveScreen` is done — see below. The rest can convert as
they are touched. Types only protect files that opt in: a `.jsx` screen reading
`session.avgFormScore.toFixed(1)` still crashes on `null` because TypeScript is
not looking at it.

Do not convert the whole app in one pass. `allowJs` means `.js` and `.tsx`
coexist indefinitely; a half-converted codebase is a normal state, not a broken
one.

**`app-state.jsx`.** It builds its context with `createContext(null)`, so
`useAppState()` infers as `never` and screens get nothing useful. The shape is
declared in `state/app-state-types.ts` and asserted at the call site; converting
that module would turn the assertion into a real check. It is the highest-value
remaining conversion for that reason.

**Web dashboard.** Same approach; it shares the REST surface and the live
observer socket. It can consume the same generated `schema.d.ts`.

**Runtime validation.** Types are a compile-time assertion, not a runtime check:
`JSON.parse(...) as ServerMessage` trusts the server. Adding a validator (Zod or
similar) at the socket boundary would close that gap, at the cost of validating
every frame at ~15 Hz. Worth revisiting if a malformed frame ever ships.
