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
import java.util.stream.IntStream
import kotlin.system.measureTimeMillis

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun main(args: Array<String>) {
    println("OMAB!")

    // Main configuration
    val configuration = Configuration(
            arms = 2,
            rewards = doubleArrayOf(1.0, 1.0, 1.0),
            horizon = 90,
            experimentProbabilities = generateUniformProbabilities(resolution = 10, count = 2),
//                        experimentProbabilities = listOf(doubleArrayOf(0.9, 0.5, 0.1)),
            iterations = 8  )

    configuration[BETA_SAMPLE_COUNT] = 100
    configuration[DISCOUNT] = 1.0

    val results: MutableList<Result> = Collections.synchronizedList(ArrayList())

    println("Number of probabilities: ${configuration.experimentProbabilities.size}")
    configuration.experimentProbabilities.forEach { println("${it[0]} ${it[1]} ") }

//    evaluateAlgorithm("OnlineValueIteration", ::onlineValueIteration, horizon, results, iterations, configuration)

//    evaluateAlgorithm("UCT", ::uct, horizon, mdp, results)
//    evaluateAlgorithm("ValueIteration", ::executeValueIteration, results, configuration)
//    evaluateAlgorithm("UCB2", ::executeUcb, results, configuration)
//    evaluateAlgorithm("UCB SS", ::executeUcb, results, configurationSS)
//    evaluateAlgorithm("Thompson Sampling", ::executeThompsonSampling, results, configuration)

//    configuration.ignoreInconsistentState = true

//    evaluateAlgorithm("UCB", ::executeUcb, results, configuration)
//    evaluateAlgorithm("Thompson Sampling", ::executeThompsonSampling, results, configuration)
    executeAlgorithm2("UCB", ::evaluateStochasticAlgorithm, ::upperConfidenceBounds, results, configuration)
    executeAlgorithm2("TS", ::evaluateStochasticAlgorithm, ::thompsonSampling, results, configuration)
    executeAlgorithm2("Gittins", ::evaluateStochasticAlgorithm, ::gittinsIndex, results, configuration)

//    configuration[LOOKAHEAD] = 1
//    executeAlgorithm2("Optimistic - l1 b1000", ::evaluateStochasticAlgorithm, ::optimisticLookahead, results, configuration)
//
//
    configuration[CONSTRAINED_PROBABILITIES] = false
//    configuration[LOOKAHEAD] = 1
//    executeAlgorithm2("UCB-Index - l${configuration[LOOKAHEAD]} b${configuration[BETA_SAMPLE_COUNT]}", ::evaluateStochasticAlgorithm, ::ucbLookahead, results, configuration)
//    configuration[LOOKAHEAD] = 2
//    executeAlgorithm2("UCB-Index - l${configuration[LOOKAHEAD]} b${configuration[BETA_SAMPLE_COUNT]}", ::evaluateStochasticAlgorithm, ::ucbLookahead, results, configuration)

    configuration[LOOKAHEAD] = 1
    executeAlgorithm2("UCB Direct - l${configuration[LOOKAHEAD]}", ::evaluateStochasticAlgorithm, ::ucbIndex, results, configuration)

    configuration[LOOKAHEAD] = 1
    configuration[DISCOUNT] = 0.0
    executeAlgorithm2("UCB-Index-d0 - l${configuration[LOOKAHEAD]} b${configuration[BETA_SAMPLE_COUNT]}", ::evaluateStochasticAlgorithm, ::ucbLookahead, results, configuration)

//    configuration[CONSTRAINED_PROBABILITIES] = true
    configuration[LOOKAHEAD] = 1
    executeAlgorithm2("UCB Direct-d0 - l${configuration[LOOKAHEAD]}", ::evaluateStochasticAlgorithm, ::ucbIndex, results, configuration)

//    configuration[DISCOUNT] = 1.0
//    configuration[LOOKAHEAD] = 2
//    executeAlgorithm2("Optimistic - l2 b200 df1", ::evaluateStochasticAlgorithm, ::optimisticLookahead, results, configuration)


//    evaluateAlgorithm("Thompson Sampling SS", ::executeThompsonSampling, results, configurationSS)
//    evaluateAlgorithm("Greedy", ::expectationMaximization, results, configuration)
//    evaluateAlgorithm("RTDP", ::executeRtdp, results, configuration)
//    evaluateAlgorithm("BRTDP", ::executeBrtdp, results, configuration)

    if (args.isNotEmpty()) {
        File(args[0]).bufferedWriter().use { results.toJson(it) }
    }
}

private fun executeAlgorithm2(
        name: String,
        executor: (world: Simulator, simulator: Simulator, DoubleArray, Configuration, algorithm: (BeliefState, Configuration, Random) -> Int, name: String) -> List<Result>,
        algorithm: (BeliefState, Configuration, Random) -> Int,
        results: MutableList<Result>,
        configuration: Configuration) {

    val experimentProbabilities = configuration.experimentProbabilities

    println("Executing $name")
    val progressBar = ProgressBar(experimentProbabilities.size)

    // Measure the execution time of the parallel execution
    val executionTime = measureTimeMillis {
        experimentProbabilities
                .parallelStream()
                .forEach { probabilities ->
                    results.addAll(executor(
                            BanditWorld(probabilities, configuration.rewards),
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

private fun evaluateAlgorithm(algorithm: String,
                              function: (Simulator, Simulator, DoubleArray, Configuration) -> List<Result>,
                              results: MutableList<Result>,
                              configuration: Configuration) {

    val executionTime = measureTimeMillis {
        executeAlgorithm(results, function, configuration)
    }

    println("$algorithm executionTime:$executionTime[ms]")
}

private fun executeAlgorithm(results: MutableList<Result>,
                             algorithm: (world: Simulator, simulator: Simulator, DoubleArray, Configuration) -> List<Result>,
                             configuration: Configuration) {

    val experimentProbabilities = configuration.experimentProbabilities

//    experimentProbabilities.forEach { print(it[0]); print(","); println(it[1]) }

    IntStream.range(0, experimentProbabilities.size)
            .parallel()
            .forEach {
                results.addAll(algorithm(
                        BanditWorld(configuration.experimentProbabilities[it], configuration.rewards),
                        BanditSimulator(configuration.rewards),
                        experimentProbabilities[it],
                        configuration))
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
//            val max = 1.1 // Uniform probabilities
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
