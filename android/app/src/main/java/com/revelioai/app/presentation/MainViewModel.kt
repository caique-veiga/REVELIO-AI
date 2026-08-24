package com.revelioai.app.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.revelioai.app.R
import com.revelioai.app.data.RevelioRepositoryImpl
import com.revelioai.app.domain.usecase.AskQuestionUseCase
import com.revelioai.app.domain.usecase.CreateSceneUseCase
import com.revelioai.app.network.ApiClient
import com.revelioai.app.network.ApiError
import com.revelioai.app.network.ApiResult
import com.revelioai.app.presentation.state.AppState
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Dona da máquina de estados inteira do app. Não conhece Context, Camera,
 * TTS ou SpeechRecognizer — só sabe orquestrar os use cases e expor estado
 * via StateFlow. `scene_id`/`conversation_id` vivem só na memória desta
 * instância (nunca persistidos), e são substituídos a cada nova foto.
 */
class MainViewModel(
    private val createScene: CreateSceneUseCase,
    private val askQuestion: AskQuestionUseCase,
) : ViewModel() {

    private val _state = MutableStateFlow<AppState>(AppState.Idle)
    val state: StateFlow<AppState> = _state.asStateFlow()

    private val _answerText = MutableStateFlow<String?>(null)
    val answerText: StateFlow<String?> = _answerText.asStateFlow()

    private var conversationId: String? = null

    /** Pra onde `retry()` deve voltar — atualizado toda vez que entramos em Error. */
    private var safeState: AppState = AppState.Idle

    fun onCaptureStarted() {
        _state.value = AppState.Capturing
    }

    fun onCaptureCancelled() {
        _state.value = readyOrIdle()
    }

    fun onCameraPermissionDenied(permanentlyDenied: Boolean) {
        enterError(
            if (permanentlyDenied) {
                R.string.error_camera_permission_permanently_denied
            } else {
                R.string.error_camera_permission_denied
            },
            fallback = readyOrIdle(),
        )
    }

    fun onMicrophonePermissionDenied(permanentlyDenied: Boolean) {
        enterError(
            if (permanentlyDenied) {
                R.string.error_microphone_permission_permanently_denied
            } else {
                R.string.error_microphone_permission_denied
            },
            fallback = readyOrIdle(),
        )
    }

    /** Nenhum app de câmera instalado para responder ao intent de captura. */
    fun onCameraUnavailable() {
        enterError(R.string.error_camera_unavailable, fallback = readyOrIdle())
    }

    /** Nenhum app de reconhecimento de voz instalado para responder ao intent. */
    fun onMicrophoneUnavailable() {
        enterError(R.string.error_microphone_unavailable, fallback = readyOrIdle())
    }

    fun onPhotoCaptured(imageBytes: ByteArray) {
        _state.value = AppState.Uploading
        viewModelScope.launch {
            val filename = "${UUID.randomUUID()}.jpg"
            when (val result = createScene(imageBytes, filename)) {
                is ApiResult.Success -> {
                    // Substitui qualquer conversation anterior — nova foto,
                    // novo contexto, o anterior é descartado por completo.
                    _state.value = AppState.Processing
                    conversationId = result.value.conversationId
                    _answerText.value = null
                    _state.value = AppState.Ready
                }

                is ApiResult.Failure -> {
                    enterError(messageFor(result.error), fallback = AppState.Idle)
                }
            }
        }
    }

    fun onListeningStarted() {
        _state.value = AppState.Listening
    }

    fun onSpeechResult(text: String?) {
        // Nunca envia transcrição vazia — barrado aqui, antes de qualquer
        // chamada de rede.
        val question = text?.trim()
        if (question.isNullOrEmpty()) {
            enterError(R.string.error_speech_empty, fallback = readyOrIdle())
            return
        }

        _state.value = AppState.Asking
        viewModelScope.launch {
            when (val result = askQuestion(conversationId, question)) {
                is ApiResult.Success -> {
                    _answerText.value = result.value.text
                    _state.value = AppState.Speaking
                }

                is ApiResult.Failure -> {
                    val fallback = if (result.error == ApiError.ConversationNotFound) {
                        AppState.Idle
                    } else {
                        AppState.Ready
                    }
                    enterError(messageFor(result.error), fallback = fallback)
                }
            }
        }
    }

    fun onTtsFinished() {
        _state.value = AppState.Ready
    }

    fun retry() {
        _state.value = safeState
    }

    private fun readyOrIdle(): AppState = if (conversationId != null) AppState.Ready else AppState.Idle

    private fun enterError(messageResId: Int, fallback: AppState) {
        safeState = fallback
        _state.value = AppState.Error(messageResId)
    }

    private fun messageFor(error: ApiError): Int = when (error) {
        ApiError.BadRequest -> R.string.error_bad_request
        ApiError.ImageTooLarge -> R.string.error_image_too_large
        ApiError.ConversationNotFound -> R.string.error_not_found
        ApiError.ServiceUnavailable -> R.string.error_service_unavailable
        ApiError.GatewayTimeout -> R.string.error_timeout
        ApiError.BadGateway -> R.string.error_server
        ApiError.ServerError -> R.string.error_server
        ApiError.Network -> R.string.error_network
        ApiError.Timeout -> R.string.error_timeout
        ApiError.InvalidResponse -> R.string.error_invalid_response
        ApiError.EmptyResponse -> R.string.error_empty_response
        ApiError.NoActiveConversation -> R.string.error_no_conversation
        ApiError.EmptyQuestion -> R.string.error_speech_empty
        ApiError.Unknown -> R.string.error_generic
    }

    companion object {
        fun factory(): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val repository = RevelioRepositoryImpl(ApiClient.api)
                MainViewModel(
                    createScene = CreateSceneUseCase(repository),
                    askQuestion = AskQuestionUseCase(repository),
                )
            }
        }
    }
}
