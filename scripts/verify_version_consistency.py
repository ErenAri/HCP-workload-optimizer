"""Verify package/changelog/tag version consistency.

Checks:
1. `pyproject.toml` project version exists in `CHANGELOG.md` section headers.
2. Optional release tag (e.g. `v2.0.0`) matches package version.
3. Optional `[Unreleased]` compare link points to the current version tag.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VERSION_RE = re.compile(r'^version\s*=\s*"(?P<version>\d+\.\d+\.\d+)"\s*$')
CHANGELOG_SECTION_RE = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\]")
UNRELEASED_LINK_RE = re.compile(
    r"^\[Unreleased\]: .*?/compare/v(?P<version>\d+\.\d+\.\d+)\.\.\.HEAD$",
)


def _read_package_version(pyproject_path: Path) -> str:
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        match = VERSION_RE.match(line.strip())
        if match:
            return match.group("version")
    raise ValueError(f"Could not find project version in {pyproject_path}")


def _read_changelog_versions(changelog_path: Path) -> list[str]:
    versions: list[str] = []
    for line in changelog_path.read_text(encoding="utf-8").splitlines():
        match = CHANGELOG_SECTION_RE.match(line.strip())
        if match:
            versions.append(match.group("version"))
    return versions


def _read_unreleased_link_base(changelog_path: Path) -> str | None:
    for line in changelog_path.read_text(encoding="utf-8").splitlines():
        match = UNRELEASED_LINK_RE.match(line.strip())
        if match:
            return match.group("version")
    return None


def _normalize_tag(tag: str) -> str:
    value = tag.strip()
    if value.startswith("refs/tags/"):
        value = value[len("refs/tags/") :]
    if value.startswith("v"):
        value = value[1:]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pyproject/changelog/tag version consistency.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Path to changelog file",
    )
    parser.add_argument(
        "--tag-ref",
        type=str,
        default="",
        help="Optional tag ref (vX.Y.Z or refs/tags/vX.Y.Z) to verify against package version",
    )
    parser.add_argument(
        "--check-unreleased-link",
        action="store_true",
        help="Require [Unreleased] compare link to use the current package version as base",
    )
    args = parser.parse_args()

    if not args.pyproject.exists():
        print(f"error: missing {args.pyproject}", file=sys.stderr)
        return 1
    if not args.changelog.exists():
        print(f"error: missing {args.changelog}", file=sys.stderr)
        return 1

    pkg_version = _read_package_version(args.pyproject)
    changelog_versions = _read_changelog_versions(args.changelog)

    if pkg_version not in changelog_versions:
        print(
            f"error: pyproject version {pkg_version} not found in changelog sections {changelog_versions}",
            file=sys.stderr,
        )
        return 1

    if args.tag_ref:
        normalized = _normalize_tag(args.tag_ref)
        if normalized != pkg_version:
            print(
                f"error: tag version {normalized} does not match pyproject version {pkg_version}",
                file=sys.stderr,
            )
            return 1

    if args.check_unreleased_link:
        link_base = _read_unreleased_link_base(args.changelog)
        if link_base is None:
            print("error: [Unreleased] compare link not found in changelog", file=sys.stderr)
            return 1
        if link_base != pkg_version:
            print(
                f"error: [Unreleased] compare base v{link_base} does not match package version v{pkg_version}",
                file=sys.stderr,
            )
            return 1

    print(
        "version consistency check passed",
        f"(package={pkg_version}, changelog_entries={len(changelog_versions)})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
