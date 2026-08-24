package com.revelioai.app.domain.usecase

import com.revelioai.app.data.RevelioRepository
import com.revelioai.app.domain.model.Answer
import com.revelioai.app.network.ApiError
import com.revelioai.app.network.ApiResult

class AskQuestionUseCase(private val repository: RevelioRepository) {

    suspend operator fun invoke(conversationId: String?, content: String): ApiResult<Answer> {
        if (conversationId.isNullOrEmpty()) {
            return ApiResult.Failure(ApiError.NoActiveConversation)
        }
        val trimmed = content.trim()
        if (trimmed.isEmpty()) {
            // O app já deveria ter barrado isso antes de chamar o use case
            // (nunca mandar transcrição vazia) — essa checagem é a rede de
            // segurança final, não o ponto principal de validação.
            return ApiResult.Failure(ApiError.EmptyQuestion)
        }
        return repository.askQuestion(conversationId, trimmed)
    }
}
