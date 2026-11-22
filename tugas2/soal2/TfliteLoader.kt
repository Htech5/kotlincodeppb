// Contoh alur memuat model .tflite di aplikasi Android (Kotlin).
// Tambahkan di build.gradle.kts modul app:
//   implementation("org.tensorflow:tensorflow-lite:2.16.1")
// Letakkan file transformer_int8.tflite di app/src/main/assets/
package com.kelompok8.transformer

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TfliteClassifier(context: Context, modelName: String = "transformer_int8.tflite") {

    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(loadModel(context, modelName), options)
    }

    // 1. Muat model dari assets (memory-mapped)
    private fun loadModel(ctx: Context, name: String): MappedByteBuffer {
        val fd = ctx.assets.openFd(name)
        FileInputStream(fd.fileDescriptor).channel.use { ch ->
            return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    // 2. Jalankan inferensi: input token id, output logit 2 kelas
    fun classify(tokenIds: IntArray): FloatArray {
        val input = arrayOf(tokenIds)
        val output = arrayOf(FloatArray(2))
        interpreter.run(input, output)
        return output[0]
    }

    // 3. Lepaskan sumber daya
    fun close() = interpreter.close()
}
