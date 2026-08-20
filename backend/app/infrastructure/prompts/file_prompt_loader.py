from pathlib import Path

from app.domain.protocols.prompt_loader import PromptNotFoundError


class FilePromptLoader:
    """Carrega prompts de arquivos de texto em `prompts/<category>/<name>`.

    Mecanismo propositalmente simples: lê o arquivo e mantém o resultado em
    memória (os prompts não mudam em runtime, só entre deploys).
    """

    def __init__(self, prompts_root: Path | str) -> None:
        self._root = Path(prompts_root)
        self._cache: dict[tuple[str, str], str] = {}

    def load(self, category: str, name: str) -> str:
        key = (category, name)
        if key in self._cache:
            return self._cache[key]

        path = self._root / category / name
        if not path.is_file():
            raise PromptNotFoundError(f"Prompt não encontrado: {path}")

        content = path.read_text(encoding="utf-8").strip()
        self._cache[key] = content
        return content
