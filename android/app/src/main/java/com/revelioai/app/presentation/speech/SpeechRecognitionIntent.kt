package com.revelioai.app.presentation.speech

import android.content.Intent
import android.speech.RecognizerIntent
import java.util.Locale

/** Diálogo nativo de reconhecimento de voz do Android — sem UI própria. */
fun buildSpeechRecognizerIntent(): Intent =
    Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale("pt", "BR").toString())
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
    }

/** Extrai o texto reconhecido, ou null se não houve resultado/vazio. */
fun extractRecognizedText(data: Intent?): String? {
    val results = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
    return results?.firstOrNull()?.trim()?.takeIf { it.isNotEmpty() }
}
