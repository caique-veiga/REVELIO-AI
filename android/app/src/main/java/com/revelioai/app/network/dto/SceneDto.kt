package com.revelioai.app.network.dto

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** Espelha exatamente `SceneCreationResponse` do backend (sem `object_count` — foi removido). */
@Serializable
data class SceneResponseDto(
    @SerialName("scene_id") val sceneId: String,
    @SerialName("conversation_id") val conversationId: String,
    val status: String,
)
