package com.revelioai.app.domain.usecase

import com.revelioai.app.data.RevelioRepository
import com.revelioai.app.domain.model.SceneCreated
import com.revelioai.app.network.ApiResult

class CreateSceneUseCase(private val repository: RevelioRepository) {
    suspend operator fun invoke(imageBytes: ByteArray, filename: String): ApiResult<SceneCreated> {
        return repository.createScene(imageBytes, filename)
    }
}
