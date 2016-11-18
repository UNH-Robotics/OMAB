package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.*
import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Rtdp(val mdp: MDP, val simulator: Simulator, val simulationCount: Int, val horizon: Int) {
    private var graph: MutableMap<BeliefState, BeliefState> = HashMap()

    fun rollOut(currentState: BeliefState, currentDepth: Int) {
        if (currentDepth >= horizon) return

        if (!graph.containsKey(currentState)) graph.put(currentState, currentState)
        mdp.addStates(mdp.generateStates(1, currentState))
        val stack = Stack<BeliefState>()

        (currentDepth..horizon).forEach {
            val (bestAction, qValue) = selectBestAction(currentState, mdp)
            val (nextState, reward) = simulator.transition(currentState, bestAction)
            stack.push(nextState)
        }

        while (!stack.isEmpty()) {
            var curState = stack.pop()
            if (!graph.containsKey(curState)) graph.put(curState, curState)
            curState = graph[curState]!!
            mdp.addStates(mdp.generateStates(1, curState))
            bellmanUtilityUpdate(curState, mdp)
        }
    }

    fun simulate(currentState: BeliefState, currentDepth: Int) {
        (0..simulationCount).forEach { rollOut(currentState, horizon) }

        throw UnsupportedOperationException("not implemented")
    }
}

fun rtdp(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
    var localMDP = MDP(horizon + 1)
    val simulationCount = 200
    var currentState = localMDP.startState

    val rtdp = Rtdp(localMDP, simulator, simulationCount, horizon)

    val totalReward = (0..horizon - 1).map { it ->
        rtdp.simulate(currentState, it)
        //mdp.addStates(mdp.generateStates(1, currentState))
        bellmanUtilityUpdate(currentState, localMDP)
        val (bestAction, bestReward) = selectBestAction(currentState, localMDP)
        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState
        reward.toDouble()
    }.sum()

    return totalReward
}