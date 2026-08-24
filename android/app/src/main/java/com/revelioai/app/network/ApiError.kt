package com.revelioai.app.network

/**
 * Erros mapeados 1:1 com os status HTTP reais do backend (ver
 * ConversationController/SceneController), mais alguns erros locais de
 * validação do próprio app (pergunta vazia, sem conversation ativa).
 */
sealed class ApiError {
    data object BadRequest : ApiError() // 400 — imagem inválida
    data object ImageTooLarge : ApiError() // 413
    data object ConversationNotFound : ApiError() // 404
    data object ServiceUnavailable : ApiError() // 503 — Ollama e Gemini indisponíveis
    data object GatewayTimeout : ApiError() // 504
    data object BadGateway : ApiError() // 502 — erro genérico da VLM
    data object ServerError : ApiError() // 500 e outros 5xx não mapeados
    data object Network : ApiError() // sem conexão, DNS, host inalcançável
    data object Timeout : ApiError() // timeout do próprio client HTTP
    data object InvalidResponse : ApiError() // JSON inválido / corpo inesperado
    data object EmptyResponse : ApiError() // corpo vazio/nulo numa resposta 2xx
    data object NoActiveConversation : ApiError() // conversation_id ausente no app
    data object EmptyQuestion : ApiError() // transcrição de voz vazia — nunca enviar
    data object Unknown : ApiError()
}

sealed class ApiResult<out T> {
    data class Success<T>(val value: T) : ApiResult<T>()
    data class Failure(val error: ApiError) : ApiResult<Nothing>()
}
