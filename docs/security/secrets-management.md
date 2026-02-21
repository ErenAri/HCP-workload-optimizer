# Secrets Management Policy

## Rules

- No secrets in source control.
- Use environment/secret manager injection for:
  - API keys,
  - registry credentials,
  - cloud tokens.
- Rotate production secrets every 90 days or on incident.

## CI/CD

- Use repository/organization secret store only.
- Never print secret values in logs.
- Restrict secret access by workflow scope.

## Validation

- Secret scanning runs in CI.
- Any leaked secret triggers immediate rotation and incident ticket.

