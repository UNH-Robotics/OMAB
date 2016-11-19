package edu.unh.cs.ai.omab.experiment

import java.io.Writer

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Result(val algorithm: String,
             val probabilities: DoubleArray,
             val optimalReward: Double,
             val reward: Double,
             val regret: Double,
             val regrets: List<Double>) {

    fun toJson(): String {
        return "{ \"algorithm\": \"$algorithm\", \"optimalReward\": $optimalReward, \"reward\": $reward, \"regret\": $regret, \"probabilities\": ${probabilities.toJson()}, \"regrets\": ${regrets.toJson()}}"
    }
}

fun DoubleArray.toJson(): String {
    val jsonStringBuilder = StringBuilder()
    jsonStringBuilder.append("[\n")
    take(size - 1).forEach { jsonStringBuilder.append(it).append(",\n") }
    jsonStringBuilder.append(last())
    jsonStringBuilder.append("\n]")
    return jsonStringBuilder.toString()
}

fun Iterable<Any>.toJson(): String {
    val jsonStringBuilder = StringBuilder()
    jsonStringBuilder.append("[\n")
    take(count() - 1).forEach { jsonStringBuilder.append(it).append(",\n") }
    jsonStringBuilder.append(last())
    jsonStringBuilder.append("\n]")
    return jsonStringBuilder.toString()
}

fun List<Result>.toJson(writer: Writer) {
    writer.append("[\n")
    if (isNotEmpty()) {
        take(size - 1).forEach { writer.append(it.toJson()).append(",\n") }
        writer.append(last().toJson())
    }
    writer.append("\n]")
}
