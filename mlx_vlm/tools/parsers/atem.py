"""Parse ATEM tool calls emitted by Muse models."""

import json
import re

# Match the invocation envelope, not the preceding reasoning turn. Server
# reasoning isolation must be able to remove that turn before parsing calls.
tool_call_start = "<atem:function_calls>"
tool_call_end = "</atem:function_calls>"

_INVOKE_PATTERN = re.compile(
    r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">(?P<body>.*?)' r"</atem:invoke>",
    re.DOTALL,
)
_PARAMETER_PATTERN = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>'
    r"(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


def _declared_parameter_types(tools, name):
    """Map parameter name -> declared JSON-Schema type for the named tool.

    Accepts OpenAI-shaped entries (``{"type": "function", "function": …}``)
    and bare function dicts. Returns ``{}`` when the tool or its properties
    are unknown, leaving value parsing on the legacy content-sniffing path.
    """
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        properties = (function.get("parameters") or {}).get("properties") or {}
        return {
            key: spec.get("type")
            for key, spec in properties.items()
            if isinstance(spec, dict)
        }
    return {}


def _parse_value(value: str, declared_type=None):
    """Parse JSON values while preserving ATEM's unquoted string form.

    A parameter declared ``string`` in the invoked tool's schema is returned
    verbatim: JSON-coercing it would silently retype string-typed parameters
    whose intended value is JSON text (e.g. a file-writing tool's ``content``)
    and then report the model's own string as a type violation. Every other
    declared type — and the schema-less legacy path — keeps the
    content-sniffing behavior that scalar parameters (``days: 3``) require.
    """
    if declared_type == "string":
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def parse_tool_call(text: str, tools=None):
    """Return one or more ATEM invocations in the common parser shape."""
    calls = []
    for invoke in _INVOKE_PATTERN.finditer(text):
        declared = _declared_parameter_types(tools, invoke.group("name"))
        arguments = {
            parameter.group("name"): _parse_value(
                parameter.group("value"), declared.get(parameter.group("name"))
            )
            for parameter in _PARAMETER_PATTERN.finditer(invoke.group("body"))
        }
        calls.append({"name": invoke.group("name"), "arguments": arguments})

    if not calls:
        raise ValueError("No ATEM function invocation found.")
    return calls[0] if len(calls) == 1 else calls
