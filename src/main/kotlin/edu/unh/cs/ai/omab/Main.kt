package edu.unh.cs.ai.omab

//import edu.unh.cs.ai.omab.algorithms.executeRtdp
import edu.unh.cs.ai.omab.algorithms.*
import edu.unh.cs.ai.omab.domain.BanditSimulator
import edu.unh.cs.ai.omab.domain.BanditWorld
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
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
    val iterations = 10

    val configuration = Configuration(3, doubleArrayOf(0.8, 0.2, 0.2), doubleArrayOf(1.0, 1.0, 1.0), horizon)

    val results: MutableList<Result> = Collections.synchronizedList(ArrayList())

//    evaluateAlgorithm("OnlineValueIteration", ::onlineValueIteration, horizon, results, iterations, configuration)

//    evaluateAlgorithm("UCT", ::uct, horizon, mdp, results)

//    evaluateAlgorithm("ValueIteration", ::executeValueIteration, horizon, results, iterations, configuration)
//   evaluateSingleAlgorithm("UCB once", ::executeUcb, horizon, results, iterations, configuration)
    evaluate3ArmAlgorithm("UCB", ::executeUcb, horizon, results, iterations, configuration)
    evaluate3ArmAlgorithm("Thompson Sampling", ::executeThompsonSampling, horizon, results, iterations, configuration)
//    evaluateAlgorithm("Greedy", ::expectationMaximization, horizon, results)
    evaluate3ArmAlgorithm("RTDP", ::executeRtdp, horizon, results, iterations, configuration)
//    evaluateAlgorithm("BRTDP", ::executeBrtdp, horizon, results, iterations)

    if (args.isNotEmpty()) {
        File(args[0]).bufferedWriter().use { results.toJson(it) }
    }
}

private fun executeSingleAlgorithm(results: MutableList<Result>,
                                   algorithm: (horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int, Configuration) -> List<Result>, horizon: Int, iterations: Int, configuration: Configuration) {

    results.addAll(algorithm(horizon, BanditWorld(configuration.probabilities), BanditSimulator(configuration.rewards), configuration.probabilities, iterations, configuration))
    print(results.toString())
}

private fun evaluateSingleAlgorithm(algorithm: String,
                                    function: (Int, Simulator, Simulator, DoubleArray, Int, Configuration) -> List<Result>,
                                    horizon: Int,
                                    results: MutableList<Result>,
                                    iterations: Int,
                                    configuration: Configuration) {

    val executionTime = measureTimeMillis {
        executeSingleAlgorithm(results, function, horizon, iterations, configuration)
    }

    println("$algorithm executionTime:$executionTime[ms]")
}

private fun evaluateAlgorithm(algorithm: String,
                              function: (Int, Simulator, Simulator, DoubleArray, Int, Configuration) -> List<Result>,
                              horizon: Int,
                              results: MutableList<Result>,
                              iterations: Int,
                              configuration: Configuration) {

    val executionTime = measureTimeMillis {
        executeAlgorithm(results, function, horizon, iterations, configuration)
    }

    println("$algorithm executionTime:$executionTime[ms]")
}


private fun evaluate3ArmAlgorithm(algorithm: String,
                                  function: (Int, Simulator, Simulator, DoubleArray, Int, Configuration) -> List<Result>,
                                  horizon: Int,
                                  results: MutableList<Result>,
                                  iterations: Int,
                                  configuration: Configuration) {

    val executionTime = measureTimeMillis {
        execute3ArmAlgorithm(results, function, horizon, iterations, configuration)
    }

    println("$algorithm executionTime:$executionTime[ms]")
}

private fun execute3ArmAlgorithm(results: MutableList<Result>,
                                 algorithm: (horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int, Configuration) -> List<Result>,
                                 horizon: Int, iterations: Int, configuration: Configuration) {
    DoubleStream
            .iterate(0.0, { i -> i + 0.1 })
            .limit(iterations.toLong())
            .parallel()
            .forEach { p1 ->
                DoubleStream
                        .iterate(0.0, { i -> i + 0.1 })
                        .limit(iterations.toLong())
                        .forEach { p2 ->
                            DoubleStream
                                    .iterate(0.0, { i -> i + 0.1 })
                                    .limit(iterations.toLong())
                                    .forEach { p3 ->
                                        results.addAll(algorithm(horizon, BanditWorld(doubleArrayOf(p1, p2, p3)), BanditSimulator(configuration.rewards), doubleArrayOf(p1, p2, p3), iterations, configuration))
                                    }

                        }
            }
}

private fun executeAlgorithm(results: MutableList<Result>,
                             algorithm: (horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int, Configuration) -> List<Result>,
                             horizon: Int, iterations: Int, configuration: Configuration) {
    DoubleStream
            .iterate(0.0, { i -> i + 0.1 })
            .limit(iterations.toLong())
            .parallel()
            .forEach { p1 ->
                DoubleStream
                        .iterate(0.0, { i -> i + 0.1 })
                        .limit(iterations.toLong())
                        .forEach { p2 ->
                            results.addAll(algorithm(horizon, BanditWorld(configuration.probabilities), BanditSimulator(configuration.rewards), doubleArrayOf(p1, p2), iterations, configuration))
                        }
            }
}