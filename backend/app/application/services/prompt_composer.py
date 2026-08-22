from app.domain.protocols.prompt_loader import PromptLoader
from app.domain.services.question_classifier import QuestionClassifier

_SYSTEM_PROMPT_NAME = "visual_assistant_v1.txt"
_SCENE_PROMPT_NAME = "scene_description_v1.txt"


class PromptComposer:
    """Monta o texto de sistema enviado à VLM (system + scene + question) e
    determina o `prompt_version` correspondente à pergunta atual.

    O prompt de pergunta (general/spatial/color/object/uncertainty) é
    selecionado por `QuestionClassifier`; system e scene são fixos.
    """

    def __init__(
        self, prompt_loader: PromptLoader, question_classifier: QuestionClassifier
    ) -> None:
        self._prompt_loader = prompt_loader
        self._question_classifier = question_classifier

    def build(self, question: str) -> tuple[str, str]:
        system_prompt = self._prompt_loader.load("system", _SYSTEM_PROMPT_NAME)
        scene_prompt = self._prompt_loader.load("scene", _SCENE_PROMPT_NAME)

        question_type = self._question_classifier.classify(question)
        question_prompt_name = f"{question_type.value}_v1.txt"
        question_prompt = self._prompt_loader.load("question", question_prompt_name)

        combined = "\n\n".join([system_prompt, scene_prompt, question_prompt])
        prompt_version = f"{question_type.value}_v1"
        return combined, prompt_version
