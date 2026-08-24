package com.revelioai.app.presentation.camera

import android.content.Context
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.util.UUID

/**
 * Cria um arquivo temporário em cache interno pra receber a foto da câmera
 * do sistema via [androidx.activity.result.contract.ActivityResultContracts.TakePicture].
 * Nunca fica em galeria pública — deletado logo após o upload.
 */
object PhotoCaptureFile {

    fun create(context: Context): Pair<Uri, File> {
        val dir = File(context.cacheDir, "captured_photos").apply { mkdirs() }
        val file = File(dir, "${UUID.randomUUID()}.jpg")
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        return uri to file
    }
}
