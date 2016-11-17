package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.expectationMaximization
import edu.unh.cs.ai.omab.algorithms.thompsonSampling
import edu.unh.cs.ai.omab.algorithms.uct
import edu.unh.cs.ai.omab.algorithms.upperConfidenceBounds
import edu.unh.cs.ai.omab.domain.*
import edu.unh.cs.ai.omab.experiment.Result
import java.io.BufferedWriter
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.lang.Math.max
import java.util.*
import java.util.stream.DoubleStream
import kotlin.system.measureTimeMillis


/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun main(args: Array<String>) {
    println("OMAB!")

    val horizon = 10

    var averageReward = 0.0
    var executionTime: Long

//    val mdp = MDP()
//    mdp.generateStates(200)
//    println("State count: ${mdp.count} map size: ${mdp.states.size}")

//    return

//    println("Time: ${measureTimeMillis {
//        val random = Random()
//        (0..10000).forEach {
//            val leftBetaDistribution = BetaDistribution((it % 30 + 1).toDouble(), (it % 100 + 1).toDouble())
//            val leftSample = leftBetaDistribution.inverseCumulativeProbability(random.nextDouble())
//        }
//    }/10000.0}")
//
    val results: MutableList<Result> = ArrayList()
    executionTime = measureTimeMillis {
        averageReward = evaluateAlgorithm(
                { probabilities, maximumReward, reward, regret -> results.add(Result("uct", probabilities, maximumReward, reward, regret)) },
                ::uct, horizon)
    }
    println("UCT  regret: $averageReward executionTime:$executionTime[ms]")
    executionTime = measureTimeMillis {
        averageReward = evaluateAlgorithm(
                { probabilities, maximumReward, reward, regret -> results.add(Result("ucb", probabilities, maximumReward, reward, regret)) },
                ::upperConfidenceBounds, horizon)
    }
    println("UCB  regret: $averageReward executionTime:$executionTime[ms]")

    executionTime = measureTimeMillis {
        averageReward = evaluateAlgorithm(
                { probabilities, maximumReward, reward, regret -> results.add(Result("Thompson sampling", probabilities, maximumReward, reward, regret)) },
                ::thompsonSampling, horizon)
    }
    println("Thompson sampling regret: $averageReward executionTime:$executionTime[ms]")

    executionTime = measureTimeMillis {
        averageReward = evaluateAlgorithm(
                { probabilities, maximumReward, reward, regret -> results.add(Result("Greedy", probabilities, maximumReward, reward, regret)) },
                ::expectationMaximization, horizon)
    }
    println("Expectation maximization regret: $averageReward executionTime:$executionTime[ms]")

    if (args.isEmpty()) {
        println(results.toString())
    } else {

        Action.getActions()
        BufferedWriter(OutputStreamWriter(FileOutputStream(args[0]), "utf-8"))
                .use { writer -> writer.write("something") }
    }
}

private fun evaluateAlgorithm(addResult: (probabilities: List<Double>, maximumReward: Double, reward: Double, regret: Double) -> Unit,
                              algorithm: (MDP, Int, Simulator, Simulator) -> Double,
                              horizon: Int): Double {
    val banditSimulator = BanditSimulator()
    val mdp = MDP()
    val averageReward = DoubleStream
            .iterate(0.0, { i -> i + 0.04 })
            .limit(25)
            .parallel()
            .map { p1 ->
                DoubleStream
                        .iterate(0.0, { i -> i + 0.04 })
                        .limit(25)
                        .mapToLong { p2 ->
                            (0..50).map {
                                val reward = algorithm(mdp, horizon, BanditWorld(p1, p2), banditSimulator)
                                val maximumReward = max(p1, p2) * horizon
                                val regret = maximumReward - reward
                                addResult(listOf(p1, p2), maximumReward, reward, regret)
                                regret
                            }.average().toLong()
                        }.average()
                        .orElseThrow { throw RuntimeException() }
            }
            .average()
            .orElseThrow { throw RuntimeException() }
    return averageReward
}