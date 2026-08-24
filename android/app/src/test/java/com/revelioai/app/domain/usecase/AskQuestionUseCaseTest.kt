package com.revelioai.app.domain.usecase

import com.revelioai.app.domain.model.Answer
import com.revelioai.app.fakes.FakeRevelioRepository
import com.revelioai.app.network.ApiError
import com.revelioai.app.network.ApiResult
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class AskQuestionUseCaseTest {

    @Test
    fun `blank question never reaches the repository`() = runTest {
        val repository = FakeRevelioRepository()
        val useCase = AskQuestionUseCase(repository)

        val result = useCase(conversationId = "conv-1", content = "   ")

        assertEquals(ApiResult.Failure(ApiError.EmptyQuestion), result)
        assertEquals(null, repository.lastAskedContent)
    }

    @Test
    fun `missing conversation id never reaches the repository`() = runTest {
        val repository = FakeRevelioRepository()
        val useCase = AskQuestionUseCase(repository)

        val result = useCase(conversationId = null, content = "quem é essa pessoa?")

        assertEquals(ApiResult.Failure(ApiError.NoActiveConversation), result)
        assertEquals(null, repository.lastAskedContent)
    }

    @Test
    fun `valid question is trimmed and forwarded to the repository`() = runTest {
        val repository = FakeRevelioRepository()
        repository.askQuestionResult = ApiResult.Success(Answer(text = "resposta", sceneId = "scene-1"))
        val useCase = AskQuestionUseCase(repository)

        useCase(conversationId = "conv-1", content = "  o que tem aqui?  ")

        assertEquals("conv-1", repository.lastAskedConversationId)
        assertEquals("o que tem aqui?", repository.lastAskedContent)
    }
}
