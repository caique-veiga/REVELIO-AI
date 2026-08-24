package com.revelioai.app.presentation.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val RevelioColorScheme = darkColorScheme(
    background = BackgroundDark,
    surface = SurfaceDark,
    onBackground = OnBackgroundLight,
    onSurface = OnBackgroundLight,
    primary = AccentTeal,
    onPrimary = OnAccent,
    error = ErrorRed,
    onError = OnBackgroundLight,
)

/**
 * Tema único, sempre escuro/alto-contraste — o app não segue o tema do
 * sistema de propósito (acessibilidade e contraste consistente importam
 * mais aqui do que combinar com o resto do aparelho).
 */
@Composable
fun RevelioAITheme(content: @Composable () -> Unit) {
    // isSystemInDarkTheme() não é usado pra decidir cores (tema fixo), só
    // documentando que a decisão foi deliberada, não esquecida.
    isSystemInDarkTheme()
    MaterialTheme(
        colorScheme = RevelioColorScheme,
        typography = RevelioTypography,
        content = content,
    )
}
