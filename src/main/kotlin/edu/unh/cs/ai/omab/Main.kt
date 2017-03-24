package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.*
import edu.unh.cs.ai.omab.domain.BanditSimulator
import edu.unh.cs.ai.omab.domain.BanditWorld
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.ConfigurationExtras.*
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.experiment.toJson
import java.io.File
import java.lang.Math.max
import java.util.*
import kotlin.system.measureTimeMillis

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun main(args: Array<String>) {
    println("OMAB!")

    // -------------------------- Main configuration --------------------------

    val arms = 3

    val configuration = Configuration(
            arms = arms,
            rewards = doubleArrayOf(1.0, 1.0, 1.0),
            horizon = 299,
            experimentProbabilities = generateConstrainedProbabilities(resolution = 10, count = arms),
            iterations = 20)


    val results: MutableList<Result> = Collections.synchronizedList(ArrayList())

    println("Number of probabilities: ${configuration.experimentProbabilities.size}")
//    configuration.experimentProbabilities.forEach { println("${it[0]} ${it[1]} ") }

    val configuredExecutor = { name: String, algorithm: Algorithm -> executeAlgorithm(name, ::evaluateStochasticAlgorithm, algorithm, results, configuration) }

    configuration[CONSTRAINED_PROBABILITIES] = false
    configuration[BETA_SAMPLE_COUNT] = 100

    executeAlgorithm("Gittins", ::evaluateStochasticAlgorithm, ::gittinsIndex, results, configuration)
    configuredExecutor("Bayes-UCB", ::bayesUpperConfidenceBounds)
    configuredExecutor("UCB", ::upperConfidenceBounds)
    executeAlgorithm("TS", ::evaluateStochasticAlgorithm, ::thompsonSampling, results, configuration)

    intArrayOf(1).forEach {
        configuration[LOOKAHEAD] = it
        configuration[DISCOUNT] = 1.0
        configuredExecutor("Gittins-Value ${if (configuration[CONSTRAINED_PROBABILITIES] as Boolean) "Constrained" else ""}", ::gittinsValueLookahead)
        configuration[DISCOUNT] = 0.4
        configuredExecutor("UCB-Value ${if (configuration[CONSTRAINED_PROBABILITIES] as Boolean) "Constrained" else ""} - d${configuration[DISCOUNT]}", ::ucbValueLookahead)
        configuration[DISCOUNT] = 1.0
        configuredExecutor("UCB-Value ${if (configuration[CONSTRAINED_PROBABILITIES] as Boolean) "Constrained" else ""} - b${configuration[BETA_SAMPLE_COUNT]} d${configuration[DISCOUNT]}", ::ucbValueLookahead)
    }

    // ------------------------------------------------------------------------

    if (args.isNotEmpty()) {
        File(args[0]).bufferedWriter().use { results.toJson(it) }
    }
}

typealias Algorithm = (BeliefState, Configuration, Random) -> Int
typealias AlgorithmEvaluator = (world: Simulator, simulator: Simulator, Pair<Int, DoubleArray>, Configuration, algorithm: Algorithm, name: String) -> List<Result>

private fun executeAlgorithm(
        name: String,
        executor: (world: Simulator, simulator: Simulator, Pair<Int, DoubleArray>, Configuration, algorithm: Algorithm, name: String) -> List<Result>,
        algorithm: (BeliefState, Configuration, Random) -> Int,
        results: MutableList<Result>,
        configuration: Configuration) {

    val experimentProbabilities = configuration.experimentProbabilities.mapIndexed { index, doubles -> index to doubles }

    println("Executing $name")
    val progressBar = ProgressBar(experimentProbabilities.size)

    // Measure the execution time of the parallel execution
    val executionTime = measureTimeMillis {
        experimentProbabilities
                .parallelStream()
                .forEach { probabilities ->
                    results.addAll(executor(
                            BanditWorld(probabilities.second, configuration.rewards),
                            BanditSimulator(configuration.rewards),
                            probabilities,
                            configuration, algorithm, name))
                    progressBar.updateProgress()
                }
    }
    println("\n$name execution time:$executionTime[ms]\n")
}

class ProgressBar(val maxProgress: Int) {
    var currentProgress = 0
    var lock = Object()

    fun updateProgress() = synchronized(lock) {
        currentProgress++
        val ratio = currentProgress.toDouble() / maxProgress

        val builder = StringBuilder("\r|                                                                              |")
        builder.append("\r|")
        (1..78).forEach {
            builder.append(if (it / 80.0 > ratio) "" else "\u2588")
        }
        print(builder.toString())
    }
}

/**
 * Generate arm probabilities such that the probabilities with lower index are higher.
 * For example: [0.5, 0.3, 0.2]
 *
 * @param resolution is the granularity of the probabilities. It defines the bucket size of the discretization.
 * @param count is the number of arms/probabilities to generate.
 *
 * @return List of probabilities.
 */
private fun generateConstrainedProbabilities(resolution: Int, count: Int): List<DoubleArray> {
    val step = 1.0 / resolution

    fun generateLevel(max: Double): DoubleArray = DoubleArray((max / step).toInt(), { max(0.0, max - (it + 1) * step) })

    var current: MutableList<DoubleArray>
    var next = ArrayList<DoubleArray>()

    current = generateLevel(1.0 + step).map {
        val firstLevel = DoubleArray(count)
        firstLevel[0] = it
        firstLevel
    }.toMutableList()

    // Generate states level by level
    (1..count - 1).forEach { level ->
        current.forEach { ps ->
            // Get possible next level
            val max = ps[level - 1] + step // Constrained probabilities
            if (max < step * (count - level)) {
                return@forEach // Make sure that we have enough for the next levels
            }

            val probabilities = generateLevel(max)

            probabilities.forEach { p ->
                val extendesProbabilities = ps.copyOf()
                extendesProbabilities[level] = p
                next.add(extendesProbabilities)
            }
        }

        current = next
        next = ArrayList<DoubleArray>()
    }

    return current
}

private fun generateUniformProbabilities(resolution: Int, count: Int): List<DoubleArray> {
    val step = 1.0 / resolution

    val probabilities = DoubleArray(resolution) { step * it }
    var current: List<DoubleArray> = List(resolution) { DoubleArray(count) }

    current.forEachIndexed { i, doubles ->
        doubles[0] = probabilities[i]
    }

    // Generate states level by level
    (1..count - 1).forEach { level ->
        current = current.map { left ->
            probabilities.map { next ->
                val newProbability = left.copyOf()
                newProbability[level] = next
                newProbability
            }
        }.flatten()
    }

    return current
}
