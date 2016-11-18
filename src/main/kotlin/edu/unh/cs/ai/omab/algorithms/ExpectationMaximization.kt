package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.stream.IntStream


fun expectationMaximization(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
    var currentState = mdp.startState

    return IntStream.iterate(0, { i -> i + 1 }).limit(horizon.toLong()).mapToDouble {
        val (nextState, reward) = if (currentState.leftMean() > currentState.rightMean()) {
            world.transition(currentState, Action.LEFT)
        } else {
            world.transition(currentState, Action.RIGHT)
        }

        currentState = nextState

        reward
    }.sum()
}