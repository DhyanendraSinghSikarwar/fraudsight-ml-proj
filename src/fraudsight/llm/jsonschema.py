"""Turn a Pydantic model into a provider-acceptable JSON Schema.

Structured-output endpoints support a deliberately restricted subset of JSON
Schema: object/array/scalar types, `enum`, `const`, `anyOf`, `$ref`/`$defs`, and
`additionalProperties: false`. They reject the *quantitative* constraints
Pydantic emits freely -- `maxLength`, `minItems`, `minimum`, `pattern`.

So the constraints get enforced in two places, on purpose:

  shape      by the provider, via this sanitised schema
  semantics  by Pydantic, client-side, after the call

That split is what makes the repair loop meaningful. A model that returns eight
evidence items when the cap is five produces a *valid-shaped* object that fails
Pydantic validation, and the graph can hand the validation error back and ask
for a fix -- which is far more useful than a provider-side rejection with no
partial output to work from.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Quantitative / lexical constraints the structured-output subset rejects.
# Enforced by Pydantic after the call instead of by the provider during it.
_UNSUPPORTED_KEYS = frozenset(
    {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minProperties",
        "maxProperties",
        "pattern",
        "patternProperties",
        "default",
        "examples",
        "contains",
        "propertyNames",
        "dependentRequired",
    }
)

# `format` is supported, but only for this closed set.
_SUPPORTED_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)


def sanitise(node: Any) -> Any:
    """Recursively strip unsupported keywords and pin `additionalProperties`."""
    if isinstance(node, list):
        return [sanitise(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYS:
            continue
        if key == "format" and value not in _SUPPORTED_FORMATS:
            continue
        out[key] = sanitise(value)

    # Every object must close itself off, and every property must be required --
    # the subset has no notion of "optional". Nullability is expressed as a union
    # with null instead, which Pydantic's `X | None` already produces.
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    return out


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Provider-ready schema for a Pydantic model."""
    raw = model.model_json_schema(mode="serialization")
    return sanitise(raw)


def describe_violation(exc: Exception, schema_name: str) -> str:
    """Render a validation failure as a repair instruction the model can act on.

    Pydantic's default rendering includes `input_value` dumps and doc URLs that
    are noise to a model and inflate the repair prompt. This keeps the field
    path and the rule that was broken, which is all that is actionable.
    """
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return f"{schema_name} rejected the output: {exc}"

    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        lines.append(f"- field `{loc}`: {err.get('msg', 'invalid')}")
    joined = "\n".join(lines[:12])
    more = "" if len(lines) <= 12 else f"\n- ... and {len(lines) - 12} more"
    return f"The previous output failed {schema_name} validation:\n{joined}{more}"
