from app.domain.entities.tool_call import ToolDefinition

REGISTER_PERSON_TOOL = "register_person"
IDENTIFY_PERSONS_TOOL = "identify_persons"

PERSON_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name=REGISTER_PERSON_TOOL,
        description="Cadastra uma pessoa no sistema com nome, relacionamento e foto do rosto.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Nome completo da pessoa"},
                "relationship": {
                    "type": "string",
                    "description": (
                        "Relacionamento com o usuário (ex: 'minha irmã', 'do trabalho', 'amigo')"
                    ),
                },
            },
            "required": ["name", "relationship"],
        },
    ),
    ToolDefinition(
        name=IDENTIFY_PERSONS_TOOL,
        description="Identifica pessoas na foto comparando com pessoas cadastradas.",
        parameters={"type": "object", "properties": {}},
    ),
]
