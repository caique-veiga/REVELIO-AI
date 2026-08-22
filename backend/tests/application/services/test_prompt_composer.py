from app.application.services.prompt_composer import PromptComposer
from app.domain.services.question_classifier import QuestionClassifier


class FakePromptLoader:
    def __init__(self) -> None:
        self.requested: list[tuple[str, str]] = []

    def load(self, category: str, name: str) -> str:
        self.requested.append((category, name))
        return f"[{category}/{name}]"


def test_build_combines_system_scene_and_question_prompts() -> None:
    loader = FakePromptLoader()
    composer = PromptComposer(prompt_loader=loader, question_classifier=QuestionClassifier())

    text, _ = composer.build("Onde está a mochila?")

    assert "[system/visual_assistant_v1.txt]" in text
    assert "[scene/scene_description_v1.txt]" in text
    assert "[question/spatial_v1.txt]" in text


def test_build_selects_question_prompt_matching_classification() -> None:
    loader = FakePromptLoader()
    composer = PromptComposer(prompt_loader=loader, question_classifier=QuestionClassifier())

    composer.build("Qual a cor da mochila?")

    assert ("question", "color_v1.txt") in loader.requested


def test_build_returns_prompt_version_matching_question_type() -> None:
    loader = FakePromptLoader()
    composer = PromptComposer(prompt_loader=loader, question_classifier=QuestionClassifier())

    _, general_version = composer.build("O que estou vendo?")
    _, spatial_version = composer.build("Onde está a mochila?")
    _, color_version = composer.build("Qual a cor da mochila?")
    _, object_version = composer.build("Tem um cachorro na cena?")
    _, uncertainty_version = composer.build("Você tem certeza disso?")

    assert general_version == "general_v1"
    assert spatial_version == "spatial_v1"
    assert color_version == "color_v1"
    assert object_version == "object_v1"
    assert uncertainty_version == "uncertainty_v1"
