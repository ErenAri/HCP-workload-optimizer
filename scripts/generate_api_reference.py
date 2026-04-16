"""Generate a static API reference markdown page from the OpenAPI baseline.

This avoids adding an additional mkdocs plugin dependency. The generated
file lands at ``docs/api/reference.md`` and is wired into ``mkdocs.yml``
under the Operations / API section.

Re-run when ``schemas/openapi_baseline.json`` changes:

    python scripts/generate_api_reference.py

The script also exits non-zero if the generated file would change vs. the
checked-in copy, which makes it CI-safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "schemas" / "openapi_baseline.json"
OUT_PATH = PROJECT_ROOT / "docs" / "api" / "reference.md"

_METHOD_ORDER = ["get", "post", "put", "patch", "delete", "head", "options"]


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _format_schema(schema: dict, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if "$ref" in schema:
        lines.append(f"{pad}- **$ref:** `{_ref_name(schema['$ref'])}`")
        return lines
    if "type" in schema:
        type_str = schema["type"]
        if "format" in schema:
            type_str += f" ({schema['format']})"
        lines.append(f"{pad}- **type:** `{type_str}`")
    for k in ("description", "title", "default", "minimum", "maximum", "minLength", "maxLength", "enum"):
        if k in schema:
            lines.append(f"{pad}- **{k}:** `{schema[k]}`")
    if "properties" in schema:
        lines.append(f"{pad}- **properties:**")
        required = set(schema.get("required", []))
        for name, sub in schema["properties"].items():
            req_marker = " *(required)*" if name in required else ""
            lines.append(f"{pad}  - `{name}`{req_marker}")
            lines.extend(_format_schema(sub, indent + 2))
    if "items" in schema:
        lines.append(f"{pad}- **items:**")
        lines.extend(_format_schema(schema["items"], indent + 1))
    return lines


def render(spec: dict) -> str:
    info = spec.get("info", {})
    title = info.get("title", "API Reference")
    version = info.get("version", "")
    description = info.get("description", "")

    out: list[str] = [
        f"# {title}",
        "",
        f"**OpenAPI version:** `{spec.get('openapi', '?')}`  •  **Spec version:** `{version}`",
        "",
        "_This page is auto-generated from `schemas/openapi_baseline.json` by_",
        "_`scripts/generate_api_reference.py`. Do not edit by hand._",
        "",
    ]
    if description:
        out += [description, ""]

    out += ["## Endpoints", ""]
    for path in sorted(spec.get("paths", {}).keys()):
        ops = spec["paths"][path]
        out += [f"### `{path}`", ""]
        for method in _METHOD_ORDER:
            op = ops.get(method)
            if not op:
                continue
            summary = op.get("summary") or op.get("operationId") or ""
            out += [f"#### `{method.upper()}` — {summary}", ""]
            if "description" in op:
                out += [op["description"], ""]
            if "tags" in op:
                out += [f"**Tags:** {', '.join(op['tags'])}", ""]
            if "parameters" in op:
                out += ["**Parameters:**", ""]
                for p in op["parameters"]:
                    loc = p.get("in", "?")
                    name = p.get("name", "?")
                    req = " *(required)*" if p.get("required") else ""
                    desc = p.get("description", "")
                    out += [f"- `{name}` ({loc}){req} — {desc}"]
                out += [""]
            req_body = op.get("requestBody")
            if req_body:
                out += ["**Request body:**", ""]
                content = req_body.get("content", {})
                for media, body in content.items():
                    out += [f"- `{media}`"]
                    schema = body.get("schema", {})
                    if "$ref" in schema:
                        out += [f"  - schema ref: `{_ref_name(schema['$ref'])}`"]
                    else:
                        out.extend(_format_schema(schema, indent=1))
                out += [""]
            responses = op.get("responses", {})
            if responses:
                out += ["**Responses:**", ""]
                for code in sorted(responses.keys()):
                    resp = responses[code]
                    desc = resp.get("description", "")
                    out += [f"- **{code}** — {desc}"]
                    content = resp.get("content", {})
                    for media, body in content.items():
                        schema = body.get("schema", {})
                        if "$ref" in schema:
                            out += [f"  - `{media}` -> schema ref: `{_ref_name(schema['$ref'])}`"]
                out += [""]

    components = spec.get("components", {}).get("schemas", {})
    if components:
        out += ["## Schemas", ""]
        for name in sorted(components.keys()):
            out += [f"### `{name}`", ""]
            out.extend(_format_schema(components[name]))
            out += [""]

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the generated file would change.")
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rendered = render(spec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.check and OUT_PATH.exists():
        existing = OUT_PATH.read_text(encoding="utf-8")
        if existing != rendered:
            print(f"error: {OUT_PATH} is out of date; re-run scripts/generate_api_reference.py", file=sys.stderr)
            return 1
        print(f"ok: {OUT_PATH} matches OpenAPI baseline")
        return 0

    OUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
