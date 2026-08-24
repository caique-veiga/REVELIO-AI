package com.revelioai.app.presentation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.revelioai.app.R
import com.revelioai.app.presentation.state.AppState

/**
 * Tela única do app. De propósito: sem navegação, sem lista, sem
 * configurações — dois botões grandes, um texto de status (falado e
 * escrito) e um botão de "tentar novamente" quando algo falha.
 */
@Composable
fun MainScreen(
    state: AppState,
    onCaptureClick: () -> Unit,
    onAskClick: () -> Unit,
    onRetryClick: () -> Unit,
) {
    val busy = state is AppState.Capturing ||
        state is AppState.Uploading ||
        state is AppState.Processing ||
        state is AppState.Asking

    val captureDescription = stringResource(R.string.button_capture_photo_description)
    val askDescription = stringResource(R.string.button_ask_question_description)

    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                text = statusText(state),
                style = MaterialTheme.typography.headlineSmall,
                color = if (state is AppState.Error) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.onBackground
                },
                modifier = Modifier
                    .padding(bottom = 32.dp)
                    .semantics { liveRegion = LiveRegionMode.Polite },
            )

            if (state is AppState.Error) {
                Button(
                    onClick = onRetryClick,
                    modifier = Modifier.fillMaxWidth().height(72.dp),
                ) {
                    Text(stringResource(R.string.button_retry), style = MaterialTheme.typography.labelLarge)
                }
                Spacer(modifier = Modifier.height(16.dp))
            }

            Button(
                onClick = onCaptureClick,
                enabled = !busy,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(96.dp)
                    .semantics {
                        contentDescription = captureDescription
                    },
            ) {
                Text(stringResource(R.string.button_capture_photo), style = MaterialTheme.typography.labelLarge)
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = onAskClick,
                enabled = !busy && state !is AppState.Idle,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(96.dp)
                    .semantics {
                        contentDescription = askDescription
                    },
            ) {
                Text(stringResource(R.string.button_ask_question), style = MaterialTheme.typography.labelLarge)
            }
        }
    }
}

@Composable
private fun statusText(state: AppState): String = when (state) {
    AppState.Idle -> stringResource(R.string.status_idle)
    AppState.Capturing -> stringResource(R.string.status_capturing)
    AppState.Uploading -> stringResource(R.string.status_uploading)
    AppState.Processing -> stringResource(R.string.status_processing)
    AppState.Ready -> stringResource(R.string.status_ready)
    AppState.Listening -> stringResource(R.string.status_listening)
    AppState.Asking -> stringResource(R.string.status_asking)
    AppState.Speaking -> stringResource(R.string.status_speaking)
    is AppState.Error -> stringResource(state.messageResId)
}
