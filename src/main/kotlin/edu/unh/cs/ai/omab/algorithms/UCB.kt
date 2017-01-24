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
private fun upperConfidenceBounds(horizon: Int, world: Simulator, arms: Int, armRewards: DoubleArray): List<Double> {
    val mdp: MDP = MDP(numberOfActions = arms)
    mdp.setRewards(armRewards)

    var currentState: BeliefState = mdp.startState
    var augmentedState: BeliefState = currentState
    val rewards: MutableList<Double> = ArrayList(horizon)

    (0..horizon - 1).forEach {
        val upperConfidenceBoundsValues = (0..augmentedState.alphas.size - 1).map {
            upperConfidenceBoundsValue(augmentedState.actionMean(it), augmentedState.actionSum(it), augmentedState.totalSum(), 2.0)
        }.toDoubleArray()

        val bestAction = (0..upperConfidenceBoundsValues.size - 1).maxBy { upperConfidenceBoundsValues[it] }!!

        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState
        augmentedState = currentState
        rewards.add(reward)
    }

    return rewards
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

    val averageRewards = DoubleArray(configuration.horizon)
    rewardsList.forEach { rewards ->
        (0..configuration.horizon - 1).forEach {
            averageRewards[it] = rewards[it] / configuration.iterations + averageRewards[it]
        }
    }

    val averageRegrets = averageRewards.mapIndexed { level, reward -> expectedMaxReward - reward }
    var sum = 0.0
    val cumSumRegrets = averageRegrets.map {
        sum += it
        sum
    }

    results.add(Result("UCB", probabilities, expectedMaxReward, averageRegrets.last(), expectedMaxReward - averageRegrets.last(), averageRegrets, cumSumRegrets))

    return results
}