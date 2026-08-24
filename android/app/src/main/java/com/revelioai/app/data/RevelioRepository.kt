package com.revelioai.app.data

import com.revelioai.app.domain.model.Answer
import com.revelioai.app.domain.model.SceneCreated
import com.revelioai.app.network.ApiResult

interface RevelioRepository {
    suspend fun createScene(imageBytes: ByteArray, filename: String): ApiResult<SceneCreated>

    suspend fun askQuestion(conversationId: String, content: String): ApiResult<Answer>
}
