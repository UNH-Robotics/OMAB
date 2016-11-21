package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.stream.IntStream

fun expectationMaximization(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
    var currentState = mdp.startState

    return IntStream.iterate(0, { i -> i + 1 }).limit(horizon.toLong()).mapToDouble {
        val means: DoubleArray = (0..currentState.alphas.size - 1).map {
            currentState.actionMean(it)
        }.toDoubleArray()
        var bestAction = 0
        (0..means.size - 1).forEach {
            if (means[bestAction] < means[it]) {
                bestAction = it
            } else {
                bestAction = bestAction
            }
        }

        val (nextState, reward) = world.transition(currentState, bestAction)
//        val (nextState, reward) = if (currentState.leftMean() > currentState.rightMean() {
//            world.transition(currentState, Action.LEFT)
//        } else {
//            world.transition(currentState, Action.RIGHT)
//        }

        currentState = nextState

        reward
    }.sum()
}