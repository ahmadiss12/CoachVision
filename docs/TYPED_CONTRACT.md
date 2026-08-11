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
| Mobile: `client.ts`, `sessions.ts` migrated | Done |
| Mobile: remaining 9 API service files | Not started |
| Mobile: screens migrated | Not started |
| Web dashboard | Not started |

The app is still mostly JavaScript. `allowJs` is on and `checkJs` is off, so
`.js` and `.ts` sit side by side and files convert one at a time instead of in
one rewrite.

## Running it

```powershell
cd mobile
npm run typecheck      # check every call site
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

That inconsistency is why `sessions.js` used to read
`payload.exerciseId ?? payload.exercise_id` on every field — nobody was certain
which the server sent, so the client hedged everywhere. `SessionResponse` has no
snake_case properties at all, so those fallbacks could never fire. They are gone
from `sessions.ts`.

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

**The other nine API service files.** Same pattern as `sessions.ts`, one at a
time. `auth.js` and `users.js` are the interesting ones: they sit on the
snake_case part of the API, so the types will make that explicit.

**Screens.** Convert as they are touched, service layer first — that is where
wire data enters and where a wrong assumption does the most damage.

**Web dashboard.** Same approach; it shares the REST surface and the live
observer socket. It can consume the same generated `schema.d.ts`.

**Runtime validation.** Types are a compile-time assertion, not a runtime check:
`JSON.parse(...) as ServerMessage` trusts the server. Adding a validator (Zod or
similar) at the socket boundary would close that gap, at the cost of validating
every frame at ~15 Hz. Worth revisiting if a malformed frame ever ships.
