package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action.LEFT
import edu.unh.cs.ai.omab.domain.Action.RIGHT
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Result
import org.apache.commons.math3.distribution.BetaDistribution
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun thompsonSampling(horizon: Int, world: Simulator): List<Double> {
    var currentState: BeliefState = MDP().startState
    val averageRewards: MutableList<Double> = ArrayList(horizon)
    var sum = 0.0

    (0..horizon - 1).forEach { level ->
        val leftBetaDistribution = BetaDistribution(currentState.alphaLeft.toDouble(), currentState.betaLeft.toDouble())
        val rightBetaDistribution = BetaDistribution(currentState.alphaRight.toDouble(), currentState.betaRight.toDouble())

        val leftSample = leftBetaDistribution.inverseCumulativeProbability(world.random.nextDouble())
        val rightSample = rightBetaDistribution.inverseCumulativeProbability(world.random.nextDouble())

        val (nextState, reward) = if (leftSample > rightSample) {
            world.transition(currentState, LEFT)
        } else {
            world.transition(currentState, RIGHT)
        }

        currentState = nextState
        sum += reward
        averageRewards.add(sum / (level + 1.0))
    }

    return averageRewards
}

fun executeThompsonSampling(horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int): List<Result> {
    val results: MutableList<Result> = ArrayList(iterations)
    val expectedMaxReward = probabilities.max()!!

    val rewardsList = IntStream.range(0, iterations).mapToObj {
        thompsonSampling(horizon, world)
    }

    val sumOfRewards = DoubleArray(horizon)
    rewardsList.forEach { rewards ->
        (0..horizon - 1).forEach {
            sumOfRewards[it] = rewards[it] + sumOfRewards[it]
        }
    }

    val averageRewards = sumOfRewards.map { expectedMaxReward - it / iterations }

    results.add(Result("TS", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))

    return results
}