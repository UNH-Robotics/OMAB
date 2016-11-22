package edu.unh.cs.ai.omab

//import edu.unh.cs.ai.omab.algorithms.executeRtdp
import edu.unh.cs.ai.omab.algorithms.executeRtdp
import edu.unh.cs.ai.omab.algorithms.executeThompsonSampling
import edu.unh.cs.ai.omab.algorithms.executeUcb
import edu.unh.cs.ai.omab.domain.BanditSimulator
import edu.unh.cs.ai.omab.domain.BanditWorld
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.experiment.toJson
import java.io.File
import java.lang.Math.exp
import java.lang.Math.max
import java.util.*
import java.util.stream.IntStream
import kotlin.system.measureTimeMillis

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun main(args: Array<String>) {
    println("OMAB!")

    val configuration = Configuration(
            arms = 3,
            rewards = doubleArrayOf(1.0, 1.0, 1.0),
            horizon = 10,
            experimentProbabilities = generateProbabilities(50, 3),
            iterations = 10,
            specialSauce = false)

    val configurationSS = Configuration(
            arms = 3,
            rewards = doubleArrayOf(1.0, 1.0, 1.0),
            horizon = 10,
            experimentProbabilities = generateProbabilities(50, 3),
            iterations = 10,
            specialSauce = true)
    val results: MutableList<Result> = Collections.synchronizedList(ArrayList())

//    evaluateAlgorithm("OnlineValueIteration", ::onlineValueIteration, horizon, results, iterations, configuration)

//    evaluateAlgorithm("UCT", ::uct, horizon, mdp, results)
//    evaluateAlgorithm("ValueIteration", ::executeValueIteration, results, configuration)
    evaluateAlgorithm("UCB", ::executeUcb, results, configuration)
    evaluateAlgorithm("UCB SS", ::executeUcb, results, configurationSS)
    evaluateAlgorithm("Thompson Sampling", ::executeThompsonSampling, results, configuration)
    evaluateAlgorithm("Thompson Sampling SS", ::executeThompsonSampling, results, configurationSS)
//    evaluateAlgorithm("Greedy", ::expectationMaximization, results, configuration)
//    evaluateAlgorithm("RTDP", ::executeRtdp, results, configuration)
//    evaluateAlgorithm("BRTDP", ::executeBrtdp, results, configuration)

    if (args.isNotEmpty()) {
        File(args[0]).bufferedWriter().use { results.toJson(it) }
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
                        BanditWorld(configuration.experimentProbabilities[it]),
                        BanditSimulator(configuration.rewards),
                        experimentProbabilities[it],
                        configuration))
            }
}

private fun generateProbabilities(resolution: Int, count: Int): List<DoubleArray> {
    val step = 1.0 / resolution

    fun generateLevel(max: Double): DoubleArray = DoubleArray((max / step).toInt(), { max(0.0, max - (it + 1) * step) })

    var current: MutableList<DoubleArray>
    var next = ArrayList<DoubleArray>()

    current = generateLevel(1.0).map {
        val firstLevel = DoubleArray(count)
        firstLevel[0] = it
        firstLevel
    }.toMutableList()

    // Generate states level by level
    (1..count - 1).forEach { level ->
        current.forEach { ps ->
            // Get possible next level
            val max = 1.0//ps[level - 1]
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