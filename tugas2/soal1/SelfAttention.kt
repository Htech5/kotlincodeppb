import kotlin.math.exp
import kotlin.math.sqrt

typealias Matrix = Array<DoubleArray>

// Kelas abstrak: kontrak umum setiap tahap pipeline (Abstraction)
abstract class Stage(val name: String)

// 1. Tokenizer: memecah kalimat menjadi kata
class Tokenizer : Stage("Tokenizer") {
    fun tokenize(text: String): List<String> =
        text.lowercase().replace("[^a-z ]".toRegex(), "").split(" ").filter { it.isNotBlank() }
}

// 2. StopwordRemover: membuang kata umum
class StopwordRemover(private val stopwords: Set<String>) : Stage("StopwordRemover") {
    fun remove(tokens: List<String>) = tokens.filterNot { it in stopwords }
}

// 3. Embedding: kata menjadi vektor 2 angka
class Embedding(private val dictionary: Map<String, DoubleArray>) : Stage("Embedding") {
    fun embed(tokens: List<String>): Matrix = tokens.map { dictionary[it] ?: DoubleArray(2) }.toTypedArray()
}

// 4. Softmax
class Softmax : Stage("Softmax") {
    fun apply(z: DoubleArray): DoubleArray {
        val mx = z.max(); val e = z.map { exp(it - mx) }; val s = e.sum()
        return e.map { it / s }.toDoubleArray()
    }
}

// 5. SelfAttention: Q = K = V = X
class SelfAttention(private val softmax: Softmax) : Stage("SelfAttention") {
    var scores: Matrix = arrayOf(); private set
    var scaled: Matrix = arrayOf(); private set
    var weights: Matrix = arrayOf(); private set

    private fun matmul(a: Matrix, b: Matrix) =
        Array(a.size) { i -> DoubleArray(b[0].size) { j -> a[i].indices.sumOf { a[i][it] * b[it][j] } } }
    private fun transpose(a: Matrix) = Array(a[0].size) { j -> DoubleArray(a.size) { i -> a[i][j] } }

    fun apply(x: Matrix): Matrix {
        val q = x; val k = x; val v = x
        val d = x[0].size
        scores = matmul(q, transpose(k))
        scaled = scores.map { row -> row.map { it / sqrt(d.toDouble()) }.toDoubleArray() }.toTypedArray()
        weights = scaled.map { softmax.apply(it) }.toTypedArray()
        return matmul(weights, v)
    }
}

fun show(m: Matrix) = m.joinToString(", ", "[", "]") { r -> r.joinToString(", ", "[", "]") { "%.4f".format(it) } }

fun main() {
    val input = "Saya menonton film yang bagus"
    val classes = listOf("Positif", "Negatif")

    val tokenizer = Tokenizer()
    val remover = StopwordRemover(setOf("saya", "menonton", "yang"))
    val embedding = Embedding(mapOf("film" to doubleArrayOf(1.0, 0.0), "bagus" to doubleArrayOf(0.0, 1.0)))
    val softmax = Softmax()
    val attention = SelfAttention(softmax)

    val tokens = tokenizer.tokenize(input)
    val filtered = remover.remove(tokens)
    val x = embedding.embed(filtered)
    val out = attention.apply(x)
    val probs = softmax.apply(out[0])
    val pred = classes[probs.indices.maxByOrNull { probs[it] }!!]

    println("=== Hasil Self-Attention Sederhana ===")
    println("Kalimat input: $input")
    println("Token awal: $tokens")
    println("Token tanpa stopword: $filtered")
    println("X = Q = K = V: ${show(x)}")
    println("QK^T: ${show(attention.scores)}")
    println("QK^T / sqrt(d): ${show(attention.scaled)}")
    println("Bobot attention (softmax): ${show(attention.weights)}")
    println("Output (bobot x V): ${show(out)}")
    println("Probabilitas [Positif, Negatif]: ${probs.map { "%.4f".format(it) }}")
    println("Prediksi kelas: $pred")
}
