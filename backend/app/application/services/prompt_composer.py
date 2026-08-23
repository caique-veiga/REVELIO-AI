from app.domain.protocols.prompt_loader import PromptLoader

_SYSTEM_PROMPT_NAME = "tool_calling_v1.txt"


class PromptComposer:
    """Monta o system prompt enviado à VLM.

    Pipeline unificado (Ollama e Gemini): um único system prompt, sem
    scene/question prompts concatenados — o modelo decide sozinho se
    responde direto ou chama uma tool (register_person/identify_persons).
    """

    def __init__(self, prompt_loader: PromptLoader) -> None:
        self._prompt_loader = prompt_loader

    def build_system_prompt(self) -> str:
        return self._prompt_loader.load("system", _SYSTEM_PROMPT_NAME)
