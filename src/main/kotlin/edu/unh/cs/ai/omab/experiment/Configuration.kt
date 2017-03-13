package edu.unh.cs.ai.omab.experiment

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Configuration(
        val arms: Int,
        val rewards: DoubleArray,
        val horizon: Int,
        val experimentProbabilities: List<DoubleArray>,
        val iterations: Int,
        val extras: MutableMap<String, Any> = hashMapOf(),
        var ignoreInconsistentState: Boolean = false) {

    operator fun get(key: ConfigurationExtras): Any {
        return extras[key.toString()] ?: throw RuntimeException("Configuration not found: $key")
    }

    operator fun set(key: ConfigurationExtras, value: Any) {
        extras[key.toString()] = value
    }
}

enum class ConfigurationExtras {
    LOOKAHEAD, DISCOUNT, BETA_SAMPLE_COUNT, CONSTRAINED_PROBABILITIES
}