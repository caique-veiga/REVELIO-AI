package com.revelioai.app.presentation.speech

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.UUID

/** Fala a resposta do backend em pt-BR. Não é usada pelo ViewModel diretamente
 * (evita acoplar o ViewModel a um Context do Android) — é orquestrada pela
 * Activity/Composable observando o estado. */
class TextToSpeechManager(context: Context) {

    private var isReady = false
    private var pendingText: String? = null
    private var onDone: (() -> Unit)? = null
    private lateinit var tts: TextToSpeech

    init {
        tts = TextToSpeech(context.applicationContext) { status ->
            isReady = status == TextToSpeech.SUCCESS
            if (isReady) {
                val languageResult = tts.setLanguage(Locale("pt", "BR"))
                if (languageResult == TextToSpeech.LANG_MISSING_DATA ||
                    languageResult == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.w(
                        "TextToSpeechManager",
                        "pt-BR indisponível no motor de TTS deste aparelho (result=$languageResult); " +
                            "instale o pacote de voz em Idiomas > Conversão de texto em voz.",
                    )
                }
                pendingText?.let { speakInternal(it) }
                pendingText = null
            }
        }
        tts.setOnUtteranceProgressListener(
            object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) = Unit

                override fun onDone(utteranceId: String?) {
                    onDone?.invoke()
                }

                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String?) {
                    onDone?.invoke()
                }
            }
        )
    }

    fun speak(text: String, onFinished: () -> Unit) {
        onDone = onFinished
        if (isReady) {
            speakInternal(text)
        } else {
            pendingText = text
        }
    }

    private fun speakInternal(text: String) {
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, UUID.randomUUID().toString())
    }

    fun shutdown() {
        tts.stop()
        tts.shutdown()
    }
}
