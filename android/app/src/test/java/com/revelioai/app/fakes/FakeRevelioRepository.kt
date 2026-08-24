package com.revelioai.app.fakes

import com.revelioai.app.data.RevelioRepository
import com.revelioai.app.domain.model.Answer
import com.revelioai.app.domain.model.SceneCreated
import com.revelioai.app.network.ApiResult

/** Fake em memória — nenhum teste que usa isso toca em rede de verdade. */
class FakeRevelioRepository : RevelioRepository {

    var createSceneResult: ApiResult<SceneCreated> =
        ApiResult.Success(SceneCreated(sceneId = "scene-1", conversationId = "conv-1"))
    var askQuestionResult: ApiResult<Answer> =
        ApiResult.Success(Answer(text = "resposta", sceneId = "scene-1"))

    var lastAskedConversationId: String? = null
    var lastAskedContent: String? = null

    override suspend fun createScene(imageBytes: ByteArray, filename: String): ApiResult<SceneCreated> {
        return createSceneResult
    }

    override suspend fun askQuestion(conversationId: String, content: String): ApiResult<Answer> {
        lastAskedConversationId = conversationId
        lastAskedContent = content
        return askQuestionResult
    }
}
