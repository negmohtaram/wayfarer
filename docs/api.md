# API

The interactive reference is at `/docs` on a running server. This page covers the parts that page can't show: how jobs, keys, deadlines and errors behave.

## Plan jobs

`POST /api/plans` with `{"persona": {...}}` answers `202 Accepted` with a job id and a `Location` header. The plan is built in the background. `GET /api/jobs/{id}` shows the job's status (`queued`, `running`, `done` or `failed`). When it is done, `plan_id` points to `GET /api/plans/{plan_id}`.

Jobs and plans are stored in Postgres when `DATABASE_URL` is set. Without it they are kept in memory and lost on restart.

## Idempotency

Send an `Idempotency-Key` header with each new request. A repeated POST with the same key and body returns the same job instead of starting a second one. The same key with a different body gets `422` with type `idempotency_key_reused`.

## Deadlines and retries

A job that is not finished within `WAYFARER_PLAN_DEADLINE_S` seconds of the request (120 by default, queue time included) fails with `retryable: true`. To try again, send the request with a new key, because the old key keeps returning the failed job.

## Errors

| Status | `detail.type` | Meaning |
|---|---|---|
| 422 | `unsupported_destination` | No city data for this destination. `GET /api/cities` lists the supported ones. |
| 422 | `idempotency_key_reused` | The key was already used with a different request. |
| 422 | (text) | The persona failed validation. |
| 503 | `model_unavailable` | The language model provider could not be reached or refused the call. Retry after the `Retry-After` seconds. |

## Browser UI

The page at `/` still plans through the older streaming endpoint, `POST /api/plan`, so trips made there are not stored as jobs. Moving it to the job API is the next frontend change.
