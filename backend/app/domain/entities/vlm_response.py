from dataclasses import dataclass


@dataclass(frozen=True)
class VLMResponse:
    text: str
    model: str
    duration_ms: float
