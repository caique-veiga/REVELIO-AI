package com.revelioai.app.data

import com.revelioai.app.network.ApiError
import com.revelioai.app.network.ApiResult
import com.revelioai.app.network.RevelioApi
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory

/**
 * Testa o parsing e mapeamento de erro contra um servidor HTTP fake local
 * (MockWebServer) — nunca contra o backend real (ver ETAPA 15 §16).
 */
class RevelioRepositoryImplTest {

    private lateinit var server: MockWebServer
    private lateinit var repository: RevelioRepositoryImpl

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()

        val json = Json { ignoreUnknownKeys = true }
        val client = OkHttpClient.Builder()
            .readTimeout(2, TimeUnit.SECONDS)
            .callTimeout(2, TimeUnit.SECONDS)
            .build()
        val api = Retrofit.Builder()
            .baseUrl(server.url("/"))
            .client(client)
            .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
            .build()
            .create(RevelioApi::class.java)

        repository = RevelioRepositoryImpl(api)
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun `createScene parses a successful response`() = runTest {
        server.enqueue(
            MockResponse()
                .setResponseCode(201)
                .setBody("""{"scene_id":"s1","conversation_id":"c1","status":"created"}""")
        )

        val result = repository.createScene(byteArrayOf(1, 2, 3), "photo.jpg")

        assertTrue(result is ApiResult.Success)
        val value = (result as ApiResult.Success).value
        assertEquals("s1", value.sceneId)
        assertEquals("c1", value.conversationId)

        val request = server.takeRequest()
        assertEquals("POST", request.method)
        assertTrue(request.path!!.contains("/api/v1/scenes"))
    }

    @Test
    fun `askQuestion parses a successful response and sends content as JSON`() = runTest {
        server.enqueue(
            MockResponse()
                .setResponseCode(200)
                .setBody("""{"answer":"há uma cadeira","scene_id":"s1"}""")
        )

        val result = repository.askQuestion("c1", "o que tem aqui?")

        assertTrue(result is ApiResult.Success)
        assertEquals("há uma cadeira", (result as ApiResult.Success).value.text)

        val request = server.takeRequest()
        assertTrue(request.path!!.contains("/api/v1/conversations/c1/messages"))
        assertTrue(request.body.readUtf8().contains("\"content\":\"o que tem aqui?\""))
    }

    @Test
    fun `400 maps to BadRequest`() = runTest {
        server.enqueue(MockResponse().setResponseCode(400))
        assertEquals(ApiResult.Failure(ApiError.BadRequest), repository.createScene(byteArrayOf(1), "bad.jpg"))
    }

    @Test
    fun `404 maps to ConversationNotFound`() = runTest {
        server.enqueue(MockResponse().setResponseCode(404))
        assertEquals(ApiResult.Failure(ApiError.ConversationNotFound), repository.askQuestion("missing", "oi"))
    }

    @Test
    fun `413 maps to ImageTooLarge`() = runTest {
        server.enqueue(MockResponse().setResponseCode(413))
        assertEquals(ApiResult.Failure(ApiError.ImageTooLarge), repository.createScene(byteArrayOf(1), "big.jpg"))
    }

    @Test
    fun `502 maps to BadGateway`() = runTest {
        server.enqueue(MockResponse().setResponseCode(502))
        assertEquals(ApiResult.Failure(ApiError.BadGateway), repository.askQuestion("c1", "oi"))
    }

    @Test
    fun `503 maps to ServiceUnavailable`() = runTest {
        server.enqueue(MockResponse().setResponseCode(503))
        assertEquals(ApiResult.Failure(ApiError.ServiceUnavailable), repository.askQuestion("c1", "oi"))
    }

    @Test
    fun `504 maps to GatewayTimeout`() = runTest {
        server.enqueue(MockResponse().setResponseCode(504))
        assertEquals(ApiResult.Failure(ApiError.GatewayTimeout), repository.askQuestion("c1", "oi"))
    }

    @Test
    fun `500 maps to ServerError`() = runTest {
        server.enqueue(MockResponse().setResponseCode(500))
        assertEquals(ApiResult.Failure(ApiError.ServerError), repository.askQuestion("c1", "oi"))
    }

    @Test
    fun `empty 2xx body maps to InvalidResponse`() = runTest {
        // Corpo vazio não vira `body() == null` no Retrofit com o conversor
        // kotlinx.serialization — ele tenta parsear "" como JSON e lança
        // SerializationException, então cai no mesmo balde de InvalidResponse
        // que qualquer outro JSON malformado. EmptyResponse (ver ApiError)
        // fica como proteção defensiva para um `body()` nulo de verdade
        // (ex. uma futura rota com resposta 204), não é exercitado por
        // nenhuma rota atual.
        server.enqueue(MockResponse().setResponseCode(200))
        assertEquals(ApiResult.Failure(ApiError.InvalidResponse), repository.askQuestion("c1", "oi"))
    }

    @Test
    fun `invalid JSON maps to InvalidResponse`() = runTest {
        server.enqueue(MockResponse().setResponseCode(200).setBody("not json"))
        assertEquals(ApiResult.Failure(ApiError.InvalidResponse), repository.askQuestion("c1", "oi"))
    }

    @Test
    fun `no response before the client timeout maps to Timeout`() = runTest {
        server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.NO_RESPONSE))
        assertEquals(ApiResult.Failure(ApiError.Timeout), repository.askQuestion("c1", "oi"))
    }
}
