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

## Enforcement

- OpenAPI compatibility check runs in CI against baseline.
- Contract tests validate telemetry headers and error envelope.

