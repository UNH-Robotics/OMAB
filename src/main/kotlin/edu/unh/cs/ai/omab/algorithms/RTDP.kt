package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Rtdp(val simulator: Simulator, val simulationCount: Int, val horizon: Int) {
    private var graph: MutableMap<BeliefState, UCTPlanner.UCTNode> = HashMap()

    data class Node(val qValue: Double)

    fun selectAction(currentState: BeliefState, it: Int): Action {
        throw UnsupportedOperationException("not implemented")
    }
}

fun rtdp(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Long {
    val simulationCount = 50

    var currentState = mdp.startState
    val rtdp = Rtdp(simulator, simulationCount, horizon)

    return (0..horizon).map {
        val action = rtdp.selectAction(currentState, it)
        val (nextState, reward) = world.transition(currentState, action)
        currentState = nextState

        reward.toLong()
    }.sum()
}