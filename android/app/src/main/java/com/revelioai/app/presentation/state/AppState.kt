package com.revelioai.app.presentation.state

/** Os 9 estados explícitos do fluxo — não adicionar sem necessidade real. */
sealed class AppState {
    data object Idle : AppState()
    data object Capturing : AppState()
    data object Uploading : AppState()
    data object Processing : AppState()
    data object Ready : AppState()
    data object Listening : AppState()
    data object Asking : AppState()
    data object Speaking : AppState()
    data class Error(val messageResId: Int) : AppState()
}
