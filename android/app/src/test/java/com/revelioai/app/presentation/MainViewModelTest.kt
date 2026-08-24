package com.revelioai.app.presentation

import com.revelioai.app.domain.model.Answer
import com.revelioai.app.domain.model.SceneCreated
import com.revelioai.app.domain.usecase.AskQuestionUseCase
import com.revelioai.app.domain.usecase.CreateSceneUseCase
import com.revelioai.app.fakes.FakeRevelioRepository
import com.revelioai.app.network.ApiError
import com.revelioai.app.network.ApiResult
import com.revelioai.app.presentation.state.AppState
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlinx.coroutines.Dispatchers

@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModelTest {

    private val dispatcher = StandardTestDispatcher()
    private lateinit var repository: FakeRevelioRepository
    private lateinit var viewModel: MainViewModel

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
        repository = FakeRevelioRepository()
        viewModel = MainViewModel(
            createScene = CreateSceneUseCase(repository),
            askQuestion = AskQuestionUseCase(repository),
        )
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `starts in Idle`() {
        assertEquals(AppState.Idle, viewModel.state.value)
    }

    @Test
    fun `capturing a photo moves through Capturing, Uploading and Processing to Ready`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))

        viewModel.onCaptureStarted()
        assertEquals(AppState.Capturing, viewModel.state.value)

        viewModel.onPhotoCaptured(byteArrayOf(1, 2, 3))
        assertEquals(AppState.Uploading, viewModel.state.value)

        dispatcher.scheduler.advanceUntilIdle()

        assertEquals(AppState.Ready, viewModel.state.value)
    }

    @Test
    fun `a second photo replaces the previous conversation id entirely`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-2", "conv-2"))
        viewModel.onPhotoCaptured(byteArrayOf(2))
        dispatcher.scheduler.advanceUntilIdle()

        repository.askQuestionResult = ApiResult.Success(Answer("ok", "scene-2"))
        viewModel.onSpeechResult("o que tem aqui?")
        dispatcher.scheduler.advanceUntilIdle()

        assertEquals("conv-2", repository.lastAskedConversationId)
    }

    @Test
    fun `asking a question goes Listening to Asking to Speaking to Ready`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        repository.askQuestionResult = ApiResult.Success(Answer("há uma mesa", "scene-1"))
        viewModel.onListeningStarted()
        assertEquals(AppState.Listening, viewModel.state.value)

        viewModel.onSpeechResult("o que tem na mesa?")
        assertEquals(AppState.Asking, viewModel.state.value)

        dispatcher.scheduler.advanceUntilIdle()

        assertEquals(AppState.Speaking, viewModel.state.value)
        assertEquals("há uma mesa", viewModel.answerText.value)
        assertEquals("conv-1", repository.lastAskedConversationId)

        viewModel.onTtsFinished()
        assertEquals(AppState.Ready, viewModel.state.value)
    }

    @Test
    fun `blank speech result never reaches the network`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        viewModel.onSpeechResult("   ")

        assertTrue(viewModel.state.value is AppState.Error)
        assertNull(repository.lastAskedContent)
    }

    @Test
    fun `question without an active conversation fails without calling the network`() = runTest {
        viewModel.onSpeechResult("quem é essa pessoa?")
        dispatcher.scheduler.advanceUntilIdle()

        assertTrue(viewModel.state.value is AppState.Error)
        assertNull(repository.lastAskedContent)
    }

    @Test
    fun `http error while asking returns to Ready on retry, not Idle`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        repository.askQuestionResult = ApiResult.Failure(ApiError.ServiceUnavailable)
        viewModel.onSpeechResult("o que você vê?")
        dispatcher.scheduler.advanceUntilIdle()

        assertTrue(viewModel.state.value is AppState.Error)
        viewModel.retry()
        assertEquals(AppState.Ready, viewModel.state.value)
    }

    @Test
    fun `conversation not found while asking returns to Idle on retry`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        repository.askQuestionResult = ApiResult.Failure(ApiError.ConversationNotFound)
        viewModel.onSpeechResult("o que você vê?")
        dispatcher.scheduler.advanceUntilIdle()

        viewModel.retry()
        assertEquals(AppState.Idle, viewModel.state.value)
    }

    @Test
    fun `scene creation failure returns to Idle on retry`() = runTest {
        repository.createSceneResult = ApiResult.Failure(ApiError.ServerError)
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        assertTrue(viewModel.state.value is AppState.Error)
        viewModel.retry()
        assertEquals(AppState.Idle, viewModel.state.value)
    }

    @Test
    fun `cancelling capture before any photo exists returns to Idle`() {
        viewModel.onCaptureStarted()
        viewModel.onCaptureCancelled()
        assertEquals(AppState.Idle, viewModel.state.value)
    }

    @Test
    fun `cancelling capture after a scene exists returns to Ready, not Idle`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        viewModel.onCaptureStarted()
        viewModel.onCaptureCancelled()

        assertEquals(AppState.Ready, viewModel.state.value)
    }

    @Test
    fun `no camera app available reports Error and retry returns to a safe state`() {
        viewModel.onCaptureStarted()
        viewModel.onCameraUnavailable()

        assertTrue(viewModel.state.value is AppState.Error)
        viewModel.retry()
        assertEquals(AppState.Idle, viewModel.state.value)
    }

    @Test
    fun `no speech recognizer app available reports Error and retry returns to Ready`() = runTest {
        repository.createSceneResult = ApiResult.Success(SceneCreated("scene-1", "conv-1"))
        viewModel.onPhotoCaptured(byteArrayOf(1))
        dispatcher.scheduler.advanceUntilIdle()

        viewModel.onListeningStarted()
        viewModel.onMicrophoneUnavailable()

        assertTrue(viewModel.state.value is AppState.Error)
        viewModel.retry()
        assertEquals(AppState.Ready, viewModel.state.value)
    }
}
