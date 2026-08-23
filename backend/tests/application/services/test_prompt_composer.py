from app.application.services.prompt_composer import PromptComposer


class FakePromptLoader:
    def __init__(self) -> None:
        self.requested: list[tuple[str, str]] = []

    def load(self, category: str, name: str) -> str:
        self.requested.append((category, name))
        return f"[{category}/{name}]"


def test_build_system_prompt_loads_the_tool_calling_prompt() -> None:
    loader = FakePromptLoader()
    composer = PromptComposer(prompt_loader=loader)

    text = composer.build_system_prompt()

    assert text == "[system/tool_calling_v1.txt]"
    assert loader.requested == [("system", "tool_calling_v1.txt")]
