package com.revelioai.app.data

import com.revelioai.app.domain.model.Answer
import com.revelioai.app.domain.model.SceneCreated
import com.revelioai.app.network.ApiError
import com.revelioai.app.network.ApiResult
import com.revelioai.app.network.RevelioApi
import com.revelioai.app.network.dto.AskQuestionRequestDto
import java.io.IOException
import java.net.SocketTimeoutException
import kotlinx.serialization.SerializationException
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Response

class RevelioRepositoryImpl(private val api: RevelioApi) : RevelioRepository {

    override suspend fun createScene(imageBytes: ByteArray, filename: String): ApiResult<SceneCreated> {
        val body = imageBytes.toRequestBody("image/jpeg".toMediaType())
        val part = MultipartBody.Part.createFormData("file", filename, body)
        return safeCall {
            api.createScene(part).toApiResult { dto ->
                SceneCreated(sceneId = dto.sceneId, conversationId = dto.conversationId)
            }
        }
    }

    override suspend fun askQuestion(conversationId: String, content: String): ApiResult<Answer> {
        return safeCall {
            api.askQuestion(conversationId, AskQuestionRequestDto(content = content)).toApiResult { dto ->
                Answer(text = dto.answer, sceneId = dto.sceneId)
            }
        }
    }

    private suspend fun <T> safeCall(block: suspend () -> ApiResult<T>): ApiResult<T> {
        return try {
            block()
        } catch (_: SocketTimeoutException) {
            ApiResult.Failure(ApiError.Timeout)
        } catch (_: SerializationException) {
            ApiResult.Failure(ApiError.InvalidResponse)
        } catch (_: IOException) {
            ApiResult.Failure(ApiError.Network)
        } catch (_: Exception) {
            ApiResult.Failure(ApiError.Unknown)
        }
    }

    private fun <B, T> Response<B>.toApiResult(transform: (B) -> T): ApiResult<T> {
        if (!isSuccessful) {
            return ApiResult.Failure(errorForHttpCode(code()))
        }
        val responseBody = body() ?: return ApiResult.Failure(ApiError.EmptyResponse)
        return ApiResult.Success(transform(responseBody))
    }

    private fun errorForHttpCode(code: Int): ApiError = when (code) {
        400 -> ApiError.BadRequest
        404 -> ApiError.ConversationNotFound
        413 -> ApiError.ImageTooLarge
        502 -> ApiError.BadGateway
        503 -> ApiError.ServiceUnavailable
        504 -> ApiError.GatewayTimeout
        in 500..599 -> ApiError.ServerError
        else -> ApiError.Unknown
    }
}
