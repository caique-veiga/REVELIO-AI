from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolCallResponse:
    text: str | None
    tool_call: ToolCall | None
    model: str
    duration_ms: float
