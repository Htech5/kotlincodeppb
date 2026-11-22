import kotlin.math.abs
import kotlin.math.roundToInt

// Konfigurasi encoder Transformer (Encapsulation)
data class TransformerConfig(val dModel: Long, val dFf: Long, val layers: Long)

// Kelas abstrak untuk format penyimpanan bobot (Abstraction)
abstract class Precision(val name: String, val bytesPerParam: Int) {
    fun sizeBytes(params: Long) = params * bytesPerParam
}
class FP32 : Precision("FP32", 4)     // Inheritance
class INT8 : Precision("INT8", 1)

// Menghitung jumlah parameter
class ParameterCounter(private val cfg: TransformerConfig) {
    fun perLayer() = 4 * cfg.dModel * cfg.dModel + 2 * cfg.dModel * cfg.dFf
    fun total() = perLayer() * cfg.layers
}

// Kuantisasi INT8 simetris
class Int8Quantizer(values: DoubleArray) {
    val scale = values.maxOf { abs(it) } / 127.0
    fun quantize(x: Double) = (x / scale).roundToInt().coerceIn(-127, 127)
    fun dequantize(q: Int) = q * scale
}

fun main() {
    val cfg = TransformerConfig(dModel = 512, dFf = 2048, layers = 6)
    val counter = ParameterCounter(cfg)
    val total = counter.total()

    println("=== Ukuran Model Transformer ===")
    println("Parameter per layer : ${counter.perLayer()}")
    println("Parameter total     : $total")
    for (p in listOf<Precision>(FP32(), INT8())) {        // Polymorphism
        val b = p.sizeBytes(total)
        println("Ukuran ${p.name}        : $b byte (%.2f MiB)".format(b / 1048576.0))
    }

    val x = doubleArrayOf(-1.0, -0.9, 0.3, 0.8, 1.0)
    val quant = Int8Quantizer(x)
    println("\n=== Kuantisasi INT8 ===")
    println("scale = %.6f".format(quant.scale))
    for (v in x) {
        val q = quant.quantize(v); val dq = quant.dequantize(q)
        println("x = %5.1f | q = %4d | dequant = %7.4f | galat = %.4f".format(v, q, dq, abs(v - dq)))
    }
}
