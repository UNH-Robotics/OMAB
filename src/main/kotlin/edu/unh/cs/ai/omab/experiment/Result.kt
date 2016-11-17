package edu.unh.cs.ai.omab.experiment

import java.io.Writer

/**
 * @author Bence Cserna (bence@cserna.net)
 */
data class Result(val algorithm: String,
                  val probabilities: List<Double>,
                  val optimalReward: Double,
                  val reward: Double,
                  val regret: Double) {
    override fun toString(): String {
        return "{ algorithm: \"$algorithm\", optimalReward: $optimalReward, reward: $reward, regret: $regret}"
    }
}

fun List<Result>.toJson(): String {
    val jsonStringBuilder = StringBuilder()
    jsonStringBuilder.append("[\n")
    forEach { jsonStringBuilder.append(it.toString()).append(",\n") }
    jsonStringBuilder.append("\n]")
    return jsonStringBuilder.toString()
}

fun List<Result>.toJson(writer: Writer) {
    writer.append("[\n")
    forEach { writer.append(it.toString()).append(",\n") }
    writer.append("\n]")
}
