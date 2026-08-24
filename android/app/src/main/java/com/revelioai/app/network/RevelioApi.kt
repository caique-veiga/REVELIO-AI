package com.revelioai.app.network

import com.revelioai.app.network.dto.AskQuestionRequestDto
import com.revelioai.app.network.dto.AskQuestionResponseDto
import com.revelioai.app.network.dto.SceneResponseDto
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path

/**
 * As únicas 3 rotas reais do backend usadas pelo app (fora `/health`).
 * Contrato conferido diretamente contra `backend/app/api/schemas` — não
 * inventar campos aqui.
 */
interface RevelioApi {

    @Multipart
    @POST("api/v1/scenes")
    suspend fun createScene(@Part file: MultipartBody.Part): Response<SceneResponseDto>

    @POST("api/v1/conversations/{conversationId}/messages")
    suspend fun askQuestion(
        @Path("conversationId") conversationId: String,
        @Body request: AskQuestionRequestDto,
    ): Response<AskQuestionResponseDto>
}
