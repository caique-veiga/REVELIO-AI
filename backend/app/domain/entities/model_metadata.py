from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    task: str
    dataset: str
