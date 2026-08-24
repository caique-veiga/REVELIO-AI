package com.revelioai.app.network

import com.revelioai.app.BuildConfig
import java.util.concurrent.TimeUnit
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory

/**
 * Client HTTP único do app, apontando pra [BuildConfig.BACKEND_BASE_URL]
 * (nunca um IP hardcoded — configurável via local.properties, ver
 * app/build.gradle.kts).
 */
object ApiClient {

    private val json = Json { ignoreUnknownKeys = true }

    // O backend pode tentar o Ollama e, se falhar, cair pro Gemini em
    // seguida — na pior hipótese isso soma dois timeouts em sequência
    // (~30s + ~30s no backend). O timeout do client precisa ser folgado o
    // bastante pra não cortar uma resposta que o backend ainda está
    // processando de verdade.
    private const val TIMEOUT_SECONDS = 90L

    private val okHttpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .callTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .apply {
                if (BuildConfig.DEBUG) {
                    addInterceptor(
                        HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC }
                    )
                }
            }
            .build()
    }

    val api: RevelioApi by lazy {
        Retrofit.Builder()
            .baseUrl(BuildConfig.BACKEND_BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
            .build()
            .create(RevelioApi::class.java)
    }
}
