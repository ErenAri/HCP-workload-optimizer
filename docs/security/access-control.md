# Access Control Policy

## Repository and CI

- Least privilege for repository roles.
- Branch protection on `main`.
- Mandatory PR review before merge.

## Production Access

- Production access limited to on-call engineers and SRE.
- All access must be auditable.
- Emergency elevation requires incident ticket reference.

## Admin Role-Based Access Control (RBAC)

API endpoints under `/v1/admin/*` require admin-level access controlled via API key prefix:

- **Admin key prefix**: API keys must have the `admin-` prefix (e.g., `admin-production-key-12345`).
- **Non-admin keys**: rejected on admin paths with `403 FORBIDDEN`.
- **Current admin endpoints**: `POST /v1/admin/log-level` (dynamic log-level adjustment with audit logging).
- **Audit logging**: all admin operations are logged via structured audit trail recording the API key, timestamp, and old→new value.
- **Development mode**: when no API keys are configured, all paths are unrestricted.

Implementation: `python/hpcopt/api/auth.py` (`check_admin_auth()`, `ADMIN_KEY_PREFIX = "admin-"`).

## API Input Security

- **Request body size limit**: 1MB max (returns `413 PAYLOAD_TOO_LARGE`).
- **Input bounds**: Pydantic models enforce `le=`, `max_length=`, and `extra="forbid"` on all request fields.
- **RFC 7807 error responses**: all error responses use Problem Details format with `type`, `title`, `status`, `detail`, `instance` fields.

## Key Practices

- Use short-lived credentials where possible.
- Remove stale access quarterly.
- Maintain ownership matrix for privileged operations.

