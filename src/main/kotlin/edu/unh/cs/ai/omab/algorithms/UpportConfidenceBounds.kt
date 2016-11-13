package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.lang.Math.log
import java.lang.Math.sqrt
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun upperConfidenceBounds(mdp: MDP, horizon: Int, simulator: Simulator): Long {
    var currentState: BeliefState = mdp.startState

    return IntStream.iterate(0, { i -> i + 1 }).limit(horizon.toLong()).mapToLong {
        val leftQ = upperConfidenceBoundsValue(currentState.leftMean(), currentState.leftSum(), currentState.totalSum(), 2.0)
        val rightQ = upperConfidenceBoundsValue(currentState.rightMean(), currentState.leftSum(), currentState.totalSum(), 2.0)

        val (nextState, reward) = if (leftQ > rightQ) {
            simulator.transition(currentState, Action.LEFT)
        } else {
            simulator.transition(currentState, Action.RIGHT)
        }

        currentState = nextState

        return@mapToLong reward.toLong()
    }.sum()
}

fun upperConfidenceBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
    return μ + sqrt(α * log(t.toDouble()) / (2 * depth * (t - 1)))
}