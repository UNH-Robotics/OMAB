package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.executeThompsonSampling
import edu.unh.cs.ai.omab.algorithms.executeUcb
import edu.unh.cs.ai.omab.algorithms.executeValueIteration
import edu.unh.cs.ai.omab.domain.BanditSimulator
import edu.unh.cs.ai.omab.domain.BanditWorld
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.experiment.toJson
import java.io.File
import java.util.*
import java.util.stream.DoubleStream
import kotlin.system.measureTimeMillis



/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun main(args: Array<String>) {
    println("OMAB!")

    val horizon = 100
    val iterations = 200

    val results: MutableList<Result> = Collections.synchronizedList(ArrayList())

//    evaluateAlgorithm("OnlineValueIteration", ::onlineValueIteration, horizon, mdp, results)

//    evaluateAlgorithm("UCT", ::uct, horizon, mdp, results)

    evaluateAlgorithm("ValueIteration", ::executeValueIteration, horizon, results, iterations)
    evaluateAlgorithm("UCB", ::executeUcb, horizon, results, iterations)
    evaluateAlgorithm("Thompson Sampling", ::executeThompsonSampling, horizon, results, iterations)
//    evaluateAlgorithm("Greedy", ::expectationMaximization, horizon, results)
//    evaluateAlgorithm("RTDP", ::executeRtdp, horizon, results, iterations)

    if (args.isNotEmpty()) {
        File(args[0]).bufferedWriter().use { results.toJson(it) }
    }
}

private fun evaluateAlgorithm(algorithm: String,
                              function: (Int, Simulator, Simulator, DoubleArray, Int) -> List<Result>,
                              horizon: Int, results: MutableList<Result>, iterations: Int) {
    val executionTime = measureTimeMillis {
        executeAlgorithm(results, function, horizon, iterations)
    }

    println("$algorithm executionTime:$executionTime[ms]")
}

private fun executeAlgorithm(results: MutableList<Result>,
                             algorithm: (horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int) -> List<Result>,
                             horizon: Int, iterations: Int) {
    DoubleStream
            .iterate(0.0, { i -> i + 0.1 })
            .limit(10)
            .parallel()
            .forEach { p1 ->
                DoubleStream
                        .iterate(0.0, { i -> i + 0.1 })
                        .limit(10)
                        .forEach { p2 ->
                            results.addAll(algorithm(horizon, BanditWorld(p1, p2), BanditSimulator, doubleArrayOf(p1, p2), iterations))
                        }
            }
}