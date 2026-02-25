# API Versioning and Deprecation Policy

## Versioning

- Public API paths use explicit major versioning (`/v1/...`).
- Breaking changes require major version bump.
- Non-breaking additions may be released within same major version.

## Compatibility Guarantees

- Existing endpoints and methods in current major version remain available.
- Existing response fields are not removed without deprecation cycle.
- Error envelope contract remains stable:
  - `error.code`
  - `error.message`
  - `error.trace_id`

## Deprecation Window

- Minimum notice period: 90 days before removal.
- Deprecation notice published in release notes.

## Implementation

Deprecation configuration is managed by `python/hpcopt/api/deprecation.py`:
- Loads deprecated endpoint entries from `configs/api/deprecation.yaml` (cached after first load).
- Provides `set_entries_for_testing()` and `reset_for_testing()` for test isolation.
- The middleware in `api/middleware.py` adds `Deprecation`, `Sunset`, and `Link` headers for matching endpoints.

## Enforcement

- OpenAPI compatibility check runs in CI against baseline.
- Contract tests validate telemetry headers and error envelope.

