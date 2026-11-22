import kotlin.math.exp
import kotlin.math.max

// Kelas abstrak: kontrak umum setiap layer (Abstraction)
abstract class Layer(val name: String)

// 1. Convolution Layer (filter 2x2, stride 1)
class ConvolutionLayer(private val filter: Array<IntArray>, private val stride: Int = 1) : Layer("Convolution") {
    fun apply(input: Array<IntArray>): Array<IntArray> {
        val outRows = (input.size - filter.size) / stride + 1
        val outCols = (input[0].size - filter[0].size) / stride + 1
        return Array(outRows) { i ->
            IntArray(outCols) { j ->
                var sum = 0
                for (fi in filter.indices) for (fj in filter[0].indices)
                    sum += input[i * stride + fi][j * stride + fj] * filter[fi][fj]
                sum
            }
        }
    }
}

// 2. ReLU
class ReLU : Layer("ReLU") {
    fun apply(input: Array<IntArray>) =
        Array(input.size) { i -> IntArray(input[0].size) { j -> max(0, input[i][j]) } }
}

// 3. MaxPooling 2x2
class MaxPooling(private val poolSize: Int = 2, private val stride: Int = 1) : Layer("MaxPooling") {
    fun apply(input: Array<IntArray>): Array<IntArray> {
        val outRows = (input.size - poolSize) / stride + 1
        val outCols = (input[0].size - poolSize) / stride + 1
        return Array(outRows) { i ->
            IntArray(outCols) { j ->
                var m = Int.MIN_VALUE
                for (pi in 0 until poolSize) for (pj in 0 until poolSize)
                    m = max(m, input[i * stride + pi][j * stride + pj])
                m
            }
        }
    }
}

// 4. Flatten
class Flatten : Layer("Flatten") {
    fun apply(input: Array<IntArray>): IntArray = input.flatMap { it.toList() }.toIntArray()
}

// 5. Dense: memetakan vektor fitur ke 3 skor kelas (z = W·x + b)
class Dense(private val weights: Array<DoubleArray>, private val bias: DoubleArray) : Layer("Dense") {
    fun apply(x: IntArray) = DoubleArray(weights.size) { c ->
        bias[c] + weights[c].indices.sumOf { weights[c][it] * x[it] }
    }
}

// 6. Softmax
class Softmax(private val classes: List<String>) : Layer("Softmax") {
    fun apply(z: DoubleArray): Pair<String, List<Double>> {
        val mx = z.max()
        val exps = z.map { exp(it - mx) }
        val probs = exps.map { it / exps.sum() }
        return Pair(classes[probs.indices.maxByOrNull { probs[it] }!!], probs)
    }
}

fun printMatrix(m: Array<IntArray>, title: String) {
    println(title); m.forEach { println(it.joinToString(", ", "[", "]")) }; println()
}

fun main() {
    val inputKucing = arrayOf(intArrayOf(1,1,0,0), intArrayOf(1,1,0,0), intArrayOf(0,0,1,1), intArrayOf(0,0,1,1))
    val inputAnjing = arrayOf(intArrayOf(0,1,1,0), intArrayOf(0,1,1,0), intArrayOf(0,0,0,0), intArrayOf(1,1,1,1))
    val inputBurung = arrayOf(intArrayOf(1,0,1,0), intArrayOf(0,1,0,1), intArrayOf(1,0,1,0), intArrayOf(0,1,0,1))
    val classes = listOf("Kucing", "Anjing", "Burung")

    val conv = ConvolutionLayer(arrayOf(intArrayOf(1, 0), intArrayOf(0, -1)))
    val relu = ReLU(); val pool = MaxPooling(); val flatten = Flatten()

    fun features(x: Array<IntArray>) = flatten.apply(pool.apply(relu.apply(conv.apply(x))))
    // Bobot & bias Dense dari prototipe kelas (nearest prototype): W = 2·mu, b = -||mu||^2
    val protos = listOf(inputKucing, inputAnjing, inputBurung).map { features(it) }
    val weights = protos.map { mu -> DoubleArray(mu.size) { 2.0 * mu[it] } }.toTypedArray()
    val bias = protos.map { mu -> 0.0 - mu.sumOf { it * it } }.toDoubleArray()
    classes.indices.forEach { i ->
        println("W_${classes[i]} = ${weights[i].joinToString(", ", "[", "]") { "%.0f".format(it) }}, b = ${"%.0f".format(bias[i] + 0.0)}")
    }

    val dense = Dense(weights, bias); val softmax = Softmax(classes)
    for ((name, input) in listOf("Kucing" to inputKucing, "Anjing" to inputAnjing, "Burung" to inputBurung)) {
        val c = conv.apply(input); val r = relu.apply(c); val p = pool.apply(r); val f = flatten.apply(p)
        val z = dense.apply(f); val (pred, probs) = softmax.apply(z)
        println("\n=== Hasil CNN Sederhana (input $name) ===\n")
        printMatrix(c, "Matriks setelah Convolution:"); printMatrix(r, "Matriks setelah ReLU:")
        printMatrix(p, "Matriks setelah MaxPooling:")
        println("Flatten: ${f.joinToString(", ", "[", "]")}")
        println("Skor Dense: ${z.map { "%.4f".format(it) }}")
        println("Probabilitas: ${probs.map { "%.4f".format(it) }}")
        println("Prediksi kelas: $pred")
    }
}
