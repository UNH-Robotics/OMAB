package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.lang.Math.sqrt
import java.lang.Math.pow
import java.util.*
import java.util.stream.IntStream
import kotlin.system.measureTimeMillis

/**
 * @author Bence Cserna (bence@cserna.net)
 */

fun calculateQ(state: BeliefState, action: Action, mdp: MDP): Double {
    val successProbabily = state.actionMean(action)
    val failProbability = 1 - successProbabily

    val successState = state.nextState(action, true)
    val failState = state.nextState(action, false)

    val successorLevel = state.totalSum() - 4 + 1// 4 is the sum of priors for 2 arms
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

//fun onlineValueIteration(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
//    val lookAhead: Int = 10
//    val onlineMDP: MDP = MDP(horizon + lookAhead)
//
//    var currentState: BeliefState = onlineMDP.startState
//
//    val addStartState: ArrayList<BeliefState> = ArrayList<BeliefState>()
//    addStartState.add(currentState)
//    onlineMDP.addStates(addStartState)
//
//
//    return IntStream.range(0, horizon).mapToDouble {
//        (1..(lookAhead)).forEach {
//            val generatedDepthStates: ArrayList<BeliefState> = mdp.generateStates(it, currentState)
//            onlineMDP.addStates(generatedDepthStates)
//        }
//
//        (horizon - 1 downTo 0).forEach {
//            mdp.getStates(it).forEach { bellmanUtilityUpdate(it, onlineMDP) }
//        }
//
//        val (bestAction, qValue) = selectBestAction(currentState, onlineMDP)
//        val (nextState, reward) = world.transition(currentState, bestAction)
//        currentState = nextState
//        reward.toDouble()
//    }.sum()
//}

fun calculateLookAhead(mdp: MDP, horizon: Int, world: Simulator,
                       simulator: Simulator, numberOfStates: Double): Double {

    val numberOfStatesGivenDepth =  (6.0*numberOfStates + 11.0 * (numberOfStates * numberOfStates) +
            6 * (numberOfStates * numberOfStates * numberOfStates) +
            (numberOfStates * numberOfStates * numberOfStates * numberOfStates))/24
    val depthGivenStates: Int = (1 / 2 * (-3 + sqrt(5 - 4 * (sqrt(24 * numberOfStates.toDouble() + 1))))).toInt()
    return numberOfStatesGivenDepth
}

//fun simpleValueIteration(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
//
//    val localMDP = MDP(horizon)
//
//    (0..horizon).forEach {
//        val genereatedDepthStates: ArrayList<BeliefState> = localMDP.generateStates(it, localMDP.startState)
//        localMDP.addStates(genereatedDepthStates)
//    }
//
//    // Back up values
//    (horizon - 1 downTo 0).forEach {
//        localMDP.getStates(it).forEach { bellmanUtilityUpdate(it, localMDP) }
//    }
//
//    var currentState = localMDP.startState
//    return IntStream.range(0, horizon).mapToDouble {
//        // Select action based on the policy
//        val (bestAction, qValue) = selectBestAction(currentState, localMDP)
//
//        val (nextState, reward) = world.transition(currentState, bestAction)
//        currentState = nextState
//
//        reward.toDouble()
//    }.sum()
//
//}
