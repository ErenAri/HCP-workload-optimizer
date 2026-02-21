# Access Control Policy

## Repository and CI

- Least privilege for repository roles.
- Branch protection on `main`.
- Mandatory PR review before merge.

## Production Access

- Production access limited to on-call engineers and SRE.
- All access must be auditable.
- Emergency elevation requires incident ticket reference.

## Key Practices

- Use short-lived credentials where possible.
- Remove stale access quarterly.
- Maintain ownership matrix for privileged operations.

