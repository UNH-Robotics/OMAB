package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.*
import java.util.stream.IntStream


fun expectationMaximization(mdp: MDP, horizon: Int, simulator: Simulator): Long {
    var currentState = mdp.startState

    return IntStream.iterate(0, { i -> i + 1}).limit(horizon.toLong()).mapToLong {
        val (nextState, reward) = if (currentState.leftMean() > currentState.rightMean()) {
            simulator.transition(currentState, Action.LEFT)
        } else {
            simulator.transition(currentState, Action.RIGHT)
        }

        currentState = nextState

        reward.toLong()
    }.sum()
}