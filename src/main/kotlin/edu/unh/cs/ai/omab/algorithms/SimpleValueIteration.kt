package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */

fun calculateQ(state: BeliefState, action: Action, mdp: MDP): Double {
    val successProbabily = state.actionMean(action)
    val failProbability = 1 - successProbabily

    val successState = state.nextState(action, true)
    val failState = state.nextState(action, false)

    val successorLevel = state.totalSum() - 4  + 1// 4 is the sum of priors for 2 arms
    val successMdpState = mdp.getLookupState(successorLevel, successState)
    val failMdpState = mdp.getLookupState(successorLevel, failState)

    // Calculate the probability weighed future utility
    val expectedValueOfSuccess = successProbabily * (successMdpState.utility + Action.getReward(action))
    val expectedValueOfFailure = failProbability * failMdpState.utility
    return expectedValueOfSuccess + expectedValueOfFailure
}

fun selectBestAction(state: BeliefState, mdp: MDP): Pair<Action, Double> {
    var bestAction: Action? = null
    var bestQValue = Double.NEGATIVE_INFINITY

    Action.getActions().forEach {
        val qValue = calculateQ(state, it, mdp)
        if (qValue > bestQValue) {
            bestAction = it
            bestQValue = qValue
        }
    }

    return Pair(bestAction!!, bestQValue)
}

fun bellmanUtilityUpdate(state: BeliefState, mdp: MDP) {
    val (action, qValue) = selectBestAction(state, mdp)
    state.utility = qValue
}

fun simpleValueIteration(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {

    // Back up values
    (horizon - 1 downTo 0).forEach {
        mdp.getStates(it).forEach { bellmanUtilityUpdate(it, mdp) }
    }

    var currentState = mdp.startState
    return IntStream.range(0, horizon).mapToDouble {
        // Select action based on the policy
        val (bestAction, qValue) = selectBestAction(currentState, mdp)

        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState

        reward.toDouble()
    }.sum()

}
