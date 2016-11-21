package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import java.lang.Math.log
import java.lang.Math.sqrt
import java.util.*
import java.util.stream.IntStream
import kotlin.Double.Companion.POSITIVE_INFINITY

/**
 * @author Bence Cserna (bence@cserna.net)
 */
private fun upperConfidenceBounds(horizon: Int, world: Simulator, arms: Int, rewards: DoubleArray): List<Double> {
    val mdp: MDP = MDP(numberOfActions = arms)
    mdp.setRewards(rewards)
    var currentState: BeliefState = mdp.startState
    val averageRewards: MutableList<Double> = ArrayList(horizon)
    var sum = 0.0

    (0..horizon - 1).forEach { level ->

        val upperConfidenceBoundsValues = (0..currentState.alphas.size - 1).map {
            upperConfidenceBoundsValue(currentState.actionMean(it), currentState.actionSum(it), currentState.totalSum(), 2.0)
        }.toDoubleArray()

        var bestAction = 0
        (0..upperConfidenceBoundsValues.size - 1).forEach {
            if (upperConfidenceBoundsValues[bestAction] <
                    upperConfidenceBoundsValues[it]) {
                bestAction = it
            } else {
                bestAction = bestAction
            }
        }

//        val leftQ = upperConfidenceBoundsValue(currentState.leftMean(), currentState.leftSum(), currentState.totalSum(), 2.0)
//        val rightQ = upperConfidenceBoundsValue(currentState.rightMean(), currentState.rightSum(), currentState.totalSum(), 2.0)

//        val (nextState, reward) = if (leftQ > rightQ) {
//            world.transition(currentState, Action.LEFT)
//        } else {
//            world.transition(currentState, Action.RIGHT)
//        }
        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState
        sum += reward
        averageRewards.add(sum / (level + 1.0))
    }

    return averageRewards
}

fun upperConfidenceBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
    return if (t == 1) POSITIVE_INFINITY else μ + sqrt(α * log(t.toDouble()) / (2 * depth * (t - 1)))
}

fun executeUcb(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val expectedMaxReward = probabilities.max()!!

    val rewardsList = IntStream.range(0, configuration.iterations).mapToObj {
        upperConfidenceBounds(configuration.horizon, world, configuration.arms, configuration.rewards)
    }

    val sumOfRewards = DoubleArray(configuration.horizon)
    rewardsList.forEach { rewards ->
        (0..configuration.horizon - 1).forEach {
            sumOfRewards[it] = rewards[it] + sumOfRewards[it]
        }
    }

    val averageRewards = sumOfRewards.map { expectedMaxReward - it / configuration.iterations }

    results.add(Result("UCB", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))

    return results
}