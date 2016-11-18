package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.lang.Math.log
import java.lang.Math.sqrt
import java.util.stream.IntStream
import kotlin.Double.Companion.POSITIVE_INFINITY

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun upperConfidenceBounds(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
    var currentState: BeliefState = mdp.startState

    return IntStream.iterate(0, { i -> i + 1 }).limit(horizon.toLong()).mapToDouble {
        val leftQ = upperConfidenceBoundsValue(currentState.leftMean(), currentState.leftSum(), currentState.totalSum(), 2.0)
        val rightQ = upperConfidenceBoundsValue(currentState.rightMean(), currentState.leftSum(), currentState.totalSum(), 2.0)

        val (nextState, reward) = if (leftQ > rightQ) {
            world.transition(currentState, Action.LEFT)
        } else {
            world.transition(currentState, Action.RIGHT)
        }

        currentState = nextState

        reward
    }.sum()
}

fun upperConfidenceBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
    return if (t == 1) POSITIVE_INFINITY else μ + sqrt(α * log(t.toDouble()) / (2 * depth * (t - 1)))
}