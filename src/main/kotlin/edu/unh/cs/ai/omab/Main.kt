package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.*
import edu.unh.cs.ai.omab.domain.BanditSimulator
import edu.unh.cs.ai.omab.domain.BanditWorld
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.experiment.toJson
import java.io.File
import java.lang.Math.max
import java.util.*
import java.util.stream.DoubleStream
import kotlin.reflect.KFunction4
import kotlin.system.measureTimeMillis


/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun main(args: Array<String>) {
    println("OMAB!")


    val horizon = 20
    val results: MutableList<Result> = Collections.synchronizedList(ArrayList())
    val mdp = MDP(horizon) // TODO Think about parallel access

    evaluateAlgorithm("OnlineValueIteration", ::onlineValueIteration, horizon, mdp, results)

//    evaluateAlgorithm("UCT", ::uct, horizon, mdp, results)

    evaluateAlgorithm("SimpleValueIteration", ::simpleValueIteration, horizon, mdp, results)
    evaluateAlgorithm("UCB", ::upperConfidenceBounds, horizon, mdp, results)
    evaluateAlgorithm("Thompson Sampling", ::thompsonSampling, horizon, mdp, results)
    evaluateAlgorithm("Greedy", ::expectationMaximization, horizon, mdp, results)
//    evaluateAlgorithm("Value Iteration", ::valueIteration, horizon, mdp, results)
//    evaluateAlgorithm("RTDP", ::rtdp, horizon, mdp, results)

    if (args.isNotEmpty()) {
        File(args[0]).bufferedWriter().use { results.toJson(it) }
    }
}


private fun evaluateAlgorithm(algorithm: String, function: KFunction4<MDP, Int, Simulator, Simulator, Double>, horizon: Int, mdp: MDP, results: MutableList<Result>) {
    var averageRegret = 0.0
    val executionTime = measureTimeMillis {
        averageRegret = executeAlgorithm(
                { probabilities, maximumReward, reward, regret -> results.add(Result(algorithm, probabilities, maximumReward, reward, regret)) },
                function, horizon, mdp)
    }
    println("$algorithm  regret: $averageRegret executionTime:$executionTime[ms]")
}

private fun executeAlgorithm(addResult: (probabilities: List<Double>, maximumReward: Double, reward: Double, regret: Double) -> Unit,
                             algorithm: (MDP, Int, Simulator, Simulator) -> Double,
                             horizon: Int,
                             mdp: MDP): Double {
    val banditSimulator = BanditSimulator()
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