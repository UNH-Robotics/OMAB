package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Result
import java.lang.Math.log
import java.lang.Math.sqrt
import java.util.*
import java.util.stream.IntStream
import kotlin.Double.Companion.POSITIVE_INFINITY

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun upperConfidenceBounds(horizon: Int, world: Simulator, simulator: Simulator): List<Double> {
    var currentState: BeliefState = MDP().startState
    var averageRewards: MutableList<Double> = ArrayList(horizon)
    var sum = 0.0

    (0..horizon - 1).forEach { level ->
        val leftQ = upperConfidenceBoundsValue(currentState.leftMean(), currentState.leftSum(), currentState.totalSum(), 2.0)
        val rightQ = upperConfidenceBoundsValue(currentState.rightMean(), currentState.rightSum(), currentState.totalSum(), 2.0)

        val (nextState, reward) = if (leftQ > rightQ) {
            world.transition(currentState, Action.LEFT)
        } else {
            world.transition(currentState, Action.RIGHT)
        }

        currentState = nextState
        sum += reward
        averageRewards.add(sum / (level + 1.0))
    }

    return averageRewards
}

fun upperConfidenceBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
    return if (t == 1) POSITIVE_INFINITY else μ + sqrt(α * log(t.toDouble()) / (2 * depth * (t - 1)))
}

fun executeUcb(horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int): List<Result> {
    val results: MutableList<Result> = ArrayList(iterations)
    val expectedMaxReward = probabilities.max()!!

    val rewardsList = IntStream.range(0, iterations).mapToObj {
        upperConfidenceBounds(horizon, world, simulator)
    }

    val sumOfRewards = DoubleArray(horizon)
    rewardsList.forEach { rewards ->
        (0..horizon - 1).forEach {
            sumOfRewards[it] = rewards[it] + sumOfRewards[it]
        }
    }

    val averageRewards = sumOfRewards.map { expectedMaxReward - it / iterations }

    results.add(Result("ucb", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))

    return results
}