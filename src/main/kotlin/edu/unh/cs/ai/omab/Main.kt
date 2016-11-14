package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.expectationMaximization
import edu.unh.cs.ai.omab.algorithms.thompsonSampling
import edu.unh.cs.ai.omab.algorithms.uct
import edu.unh.cs.ai.omab.algorithms.upperConfidenceBounds
import edu.unh.cs.ai.omab.domain.BanditSimulator
import edu.unh.cs.ai.omab.domain.BanditWorld
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
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

//    println("Time: ${measureTimeMillis {
//        val random = Random()
//        (0..10000).forEach {
//            val leftBetaDistribution = BetaDistribution((it % 30 + 1).toDouble(), (it % 100 + 1).toDouble())
//            val leftSample = leftBetaDistribution.inverseCumulativeProbability(random.nextDouble())
//        }
//    }/10000.0}")
//
    executionTime = measureTimeMillis {
    averageReward = evaluateAlgorithm(::uct, horizon)
    }
    println("UCT  reward: $averageReward executionTime:$executionTime[ms]")
    executionTime = measureTimeMillis {
        averageReward = evaluateAlgorithm(::upperConfidenceBounds, horizon)
    }
    println("UCB  reward: $averageReward executionTime:$executionTime[ms]")

    executionTime = measureTimeMillis {
        averageReward = evaluateAlgorithm(::thompsonSampling, horizon)
    }
    println("Thompson sampling reward: $averageReward executionTime:$executionTime[ms]")

    executionTime = measureTimeMillis {
        averageReward = evaluateAlgorithm(::expectationMaximization, horizon)
    }
    println("Expectation maximization reward: $averageReward executionTime:$executionTime[ms]")
}

private fun evaluateAlgorithm(algorithm: (MDP, Int, Simulator, Simulator) -> Long, horizon: Int): Double {
    val banditSimulator = BanditSimulator()
    val averageReward = DoubleStream
            .iterate(0.0, { i -> i + 0.04 })
            .limit(25)
//            .parallel()
            .map { p1 ->
                DoubleStream
                        .iterate(0.0, { i -> i + 0.04 })
                        .limit(25)
                        .mapToLong { p2 ->
                            (0..50).map {
                                algorithm(MDP(), horizon, BanditWorld(p1, p2), banditSimulator)
                            }.average().toLong()
                        }.average()
                        .orElseThrow { throw RuntimeException() }
            }
            .average()
            .orElseThrow { throw RuntimeException() }
    return averageReward
}