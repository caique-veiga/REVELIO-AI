package com.revelioai.app.network.dto

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** Espelha `AskQuestionRequest` do backend. */
@Serializable
data class AskQuestionRequestDto(
    val content: String,
)

/** Espelha `AskQuestionResponse` do backend (só `answer` + `scene_id`). */
@Serializable
data class AskQuestionResponseDto(
    val answer: String,
    @SerialName("scene_id") val sceneId: String,
)
