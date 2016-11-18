package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.*
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun onlineValueIteration(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
    val lookAhead: Int = 10
    var realHorizon = horizon
    val onlineMDP = MDP(horizon+lookAhead)

    var currentState: BeliefState = onlineMDP.startState

    val addStartState = ArrayList<BeliefState>()
    addStartState.add(currentState)
    onlineMDP.addStates(addStartState)

    return IntStream.range(0, realHorizon).mapToDouble {
        (1..(lookAhead)).forEach {
            val generatedDepthStates: ArrayList<BeliefState> = mdp.generateStates(it, currentState)
            onlineMDP.addStates(generatedDepthStates)
        }

        (lookAhead - 1 downTo 0).forEach {
            mdp.getStates(it).forEach { bellmanUtilityUpdate(it, onlineMDP) }
        }
        realHorizon -= lookAhead
        val (bestAction, qValue) = selectBestAction(currentState, onlineMDP)
        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState
        reward
    }.sum()

}

fun simpleValueIteration(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {

    val localMDP = MDP(horizon)

    (0..horizon).forEach {
        val genereatedDepthStates: ArrayList<BeliefState> = localMDP.generateStates(it, localMDP.startState)
        localMDP.addStates(genereatedDepthStates)
    }

    // Back up values
    (horizon - 1 downTo 0).forEach {
        localMDP.getStates(it).forEach { bellmanUtilityUpdate(it, localMDP) }
    }

    var currentState = localMDP.startState
    return IntStream.range(0, horizon).mapToDouble {
        // Select action based on the policy
        val (bestAction, qValue) = selectBestAction(currentState, localMDP)

        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState

        reward
    }.sum()

}
