package com.revelioai.app.presentation

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import com.revelioai.app.presentation.camera.PhotoCaptureFile
import com.revelioai.app.presentation.speech.TextToSpeechManager
import com.revelioai.app.presentation.speech.buildSpeechRecognizerIntent
import com.revelioai.app.presentation.speech.extractRecognizedText
import com.revelioai.app.presentation.state.AppState
import com.revelioai.app.presentation.theme.RevelioAITheme
import java.io.File

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels { MainViewModel.factory() }

    private lateinit var ttsManager: TextToSpeechManager
    private var pendingPhotoFile: File? = null

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                launchCamera()
            } else {
                val canAskAgain = shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
                viewModel.onCameraPermissionDenied(permanentlyDenied = !canAskAgain)
            }
        }

    private val requestMicPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                launchSpeechRecognizer()
            } else {
                val canAskAgain = shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)
                viewModel.onMicrophonePermissionDenied(permanentlyDenied = !canAskAgain)
            }
        }

    private val takePicture =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            val file = pendingPhotoFile
            pendingPhotoFile = null
            if (success && file != null && file.exists()) {
                viewModel.onPhotoCaptured(file.readBytes())
            } else {
                viewModel.onCaptureCancelled()
            }
            file?.delete()
        }

    private val recognizeSpeech =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            viewModel.onSpeechResult(extractRecognizedText(result.data))
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ttsManager = TextToSpeechManager(applicationContext)

        setContent {
            RevelioAITheme {
                val state by viewModel.state.collectAsState()
                val answer by viewModel.answerText.collectAsState()

                LaunchedEffect(state) {
                    if (state is AppState.Speaking) {
                        val text = answer
                        if (text.isNullOrEmpty()) {
                            viewModel.onTtsFinished()
                        } else {
                            ttsManager.speak(text) { viewModel.onTtsFinished() }
                        }
                    }
                }

                MainScreen(
                    state = state,
                    onCaptureClick = ::onCaptureClick,
                    onAskClick = ::onAskClick,
                    onRetryClick = viewModel::retry,
                )
            }
        }
    }

    private fun onCaptureClick() {
        if (hasPermission(Manifest.permission.CAMERA)) {
            launchCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun onAskClick() {
        if (hasPermission(Manifest.permission.RECORD_AUDIO)) {
            launchSpeechRecognizer()
        } else {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun hasPermission(permission: String): Boolean =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun launchCamera() {
        viewModel.onCaptureStarted()
        val (uri, file) = PhotoCaptureFile.create(applicationContext)
        pendingPhotoFile = file
        try {
            takePicture.launch(uri)
        } catch (_: ActivityNotFoundException) {
            pendingPhotoFile = null
            file.delete()
            viewModel.onCameraUnavailable()
        }
    }

    private fun launchSpeechRecognizer() {
        viewModel.onListeningStarted()
        try {
            recognizeSpeech.launch(buildSpeechRecognizerIntent())
        } catch (_: ActivityNotFoundException) {
            viewModel.onMicrophoneUnavailable()
        }
    }

    override fun onDestroy() {
        ttsManager.shutdown()
        super.onDestroy()
    }
}
