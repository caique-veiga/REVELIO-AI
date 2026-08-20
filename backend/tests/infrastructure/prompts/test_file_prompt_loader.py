from pathlib import Path

import pytest

from app.domain.protocols.prompt_loader import PromptNotFoundError
from app.infrastructure.prompts.file_prompt_loader import FilePromptLoader


@pytest.fixture
def prompts_root(tmp_path: Path) -> Path:
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "visual_assistant_v1.txt").write_text(
        "  Você é um assistente visual.  \n", encoding="utf-8"
    )
    return tmp_path


def test_load_returns_stripped_file_content(prompts_root: Path) -> None:
    loader = FilePromptLoader(prompts_root)

    content = loader.load("system", "visual_assistant_v1.txt")

    assert content == "Você é um assistente visual."


def test_load_caches_result(prompts_root: Path) -> None:
    loader = FilePromptLoader(prompts_root)
    first = loader.load("system", "visual_assistant_v1.txt")

    (prompts_root / "system" / "visual_assistant_v1.txt").write_text(
        "conteúdo alterado", encoding="utf-8"
    )
    second = loader.load("system", "visual_assistant_v1.txt")

    assert first == second == "Você é um assistente visual."


def test_load_raises_when_prompt_file_does_not_exist(prompts_root: Path) -> None:
    loader = FilePromptLoader(prompts_root)

    with pytest.raises(PromptNotFoundError):
        loader.load("question", "nonexistent_v1.txt")


def test_load_real_project_prompts() -> None:
    project_root = Path(__file__).resolve().parents[4]
    loader = FilePromptLoader(project_root / "prompts")

    assert loader.load("system", "visual_assistant_v1.txt")
    assert loader.load("scene", "scene_description_v1.txt")
    for name in ("general", "spatial", "color", "object", "uncertainty"):
        assert loader.load("question", f"{name}_v1.txt")
