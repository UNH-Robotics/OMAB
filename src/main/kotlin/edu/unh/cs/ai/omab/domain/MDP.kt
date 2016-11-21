package edu.unh.cs.ai.omab.domain

import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */

data class BeliefState(val alphas: IntArray, val betas: IntArray) {
    var utility = 1000.0

    fun alphaSum() = alphas.sum()
    fun betaSum() = betas.sum()
    fun totalSum() = alphaSum() + betaSum()

    fun size() = alphas.size

    override fun hashCode(): Int {
        var hashCode = 0
        alphas.forEach { hashCode = hashCode xor it; hashCode shl 1 }
        betas.forEach { hashCode = hashCode xor it; hashCode shl 1 }
        return hashCode
    }

    override fun equals(other: Any?): Boolean {
        return when (other) {
            null -> false
            !is BeliefState -> false
            else -> (other.alphas.size == alphas.size) && (other.betas.size == alphas.size) &&
                    (0..alphas.size - 1).all { alphas[it] == other.alphas[it] && betas[it] == other.betas[it] }
        }
    }

    fun nextState(action: Int, success: Boolean): BeliefState {
        val newAlphas = alphas.copyOf()
        val newBetas = betas.copyOf()
        if (success) {
            newAlphas[action] += 1
        } else {
            newBetas[action] += 1
        }
        return BeliefState(newAlphas, newBetas)
    }

    fun actionMean(action: Int): Double {
        return alphas[action].toDouble() / (actionSum(action).toDouble())
    }

    fun actionSum(action: Int): Int {
        return alphas[action] + betas[action]
    }
}

class MDP(depth: Int? = null, val numberOfActions: Int) {


    val states: MutableMap<BeliefState, BeliefState> = HashMap()
    private val mapsByLevel: Array<MutableMap<BeliefState, BeliefState>>
    private val statesByLevel: Array<MutableList<BeliefState>>
    private val rewards = doubleArrayOf(1.0, 1.0, 1.0)

    val startState = BeliefState(IntArray(numberOfActions, { 1 }), IntArray(numberOfActions, { 1 }))
    val actions = IntArray(numberOfActions, { it })

    init {
        mapsByLevel = Array<MutableMap<BeliefState, BeliefState>>(depth?.plus(1) ?: 0, { HashMap<BeliefState, BeliefState>() })
        statesByLevel = Array<MutableList<BeliefState>>(depth?.plus(1) ?: 0, { ArrayList<BeliefState>() })
    }


    fun getReward(action: Int): Double {
        return rewards[action]
    }

    fun addStates(statesToAdd: ArrayList<BeliefState>) {
        statesToAdd.forEach {
            val level = it.alphaSum() + it.betaSum() - 4
            mapsByLevel[level][it] = it
            statesByLevel[level].add(it)
            if (!states.containsKey(it)) {
                states[it] = it
            }
        }
    }

//    fun generateStates(depth: Int) {
//        val sum = depth + 4 //  4 is the prior
//        for (leftAlpha in 1..sum) {
//            for (leftBeta in 1..(sum - leftAlpha)) {
//                for (rightAlpha in 1..(sum - leftAlpha - leftBeta)) {
//                    for (rightBeta in 1..(sum - leftAlpha - leftBeta - rightAlpha)) {
//                        val state = BeliefState(leftAlpha, leftBeta, rightAlpha, rightBeta)
//                        count++
//                        states[state] = state
//                        val level = leftAlpha + leftBeta + rightAlpha + rightBeta - startState.totalSum()// sum of start is the prior
//                        mapsByLevel[level][state] = state
//                        statesByLevel[level].add(state)
//                    }
//                }
//            }
//        }
//    }

    fun generateStates(depth: Int, state: BeliefState): ArrayList<BeliefState> {
        if (numberOfActions == 2) {
            return generateStates2(depth, state)
        } else {
            TODO()
        }
    }

    fun generateStatess(depth: Int, state: BeliefState, listToFill: ArrayList<BeliefState>) {

        if(depth>3)
            return

        println(state)
        for (i in 0..numberOfActions-1){
            for(j in listOf(true, false)) {
                generateStatess(depth + 1, state.nextState(i, j), listToFill)
            }
        }

        /*if (depth >= 0) {
            (0..(numberOfActions * 2) - 1).forEach {
                (0..(((numberOfActions * 2)) / 2) - 1).forEach {
                    val alphas = state.alphas.copyOf()
                    val betas = state.betas.copyOf()
                    alphas[it] += 1
                    if (depth == 0) {
                        listToFill.add(BeliefState(alphas, state.betas))
                    }
                    generateStates(depth - 1, BeliefState(alphas, state.betas), listToFill)
                    betas[it] += 1
                    if (depth == 0) {
                        listToFill.add(BeliefState(state.alphas, betas))
                    }
                    generateStates(depth - 1, BeliefState(state.alphas, betas), listToFill)
                }
            }
        }*/
    }

    fun generateStates2(depth: Int, state: BeliefState): ArrayList<BeliefState> {
        val initializedStates = ArrayList<BeliefState>()
        for (x in 0..(depth)) {
            for (y in 0..(depth - x)) {
                (0..(depth - x - y))
                        .mapTo(initializedStates) {
                            val alphas = state.alphas.copyOf()
                            val betas = state.betas.copyOf()
                            /** alphaLeft && betaLeft*/
                            alphas[0] = x + alphas[0]
                            betas[0] = y + betas[0]
                            /** alphaRight && betaRight*/
                            alphas[1] = it + alphas[1]
                            betas[1] = depth - x - y - it + betas[1]
                            BeliefState(alphas, betas)
//                            val alphas: IntArray =
//                                    BeliefState(x + state.alphaLeft, y + state.betaLeft,
//                                            it + state.alphaRight, depth - x - y - it + state.betaRight)
                        }
            }
        }
        return initializedStates
    }

    fun getStates(level: Int): List<BeliefState> = statesByLevel[level]
    fun getLookupState(level: Int, state: BeliefState): BeliefState = mapsByLevel[level][state]
            ?: throw RuntimeException("Cannot find state: $state on level $level")

    var count = 0
}

fun calculateQ(state: BeliefState, action: Int, mdp: MDP): Double {
    val successProbability = state.actionMean(action)
    val failProbability = 1 - successProbability

    val successState = state.nextState(action, true)
    val failState = state.nextState(action, false)

    val successorLevel = state.totalSum() - mdp.startState.totalSum() + 1// the sum of priors for n arms
    val successMdpState = mdp.getLookupState(successorLevel, successState)
    val failMdpState = mdp.getLookupState(successorLevel, failState)

    // Calculate the probability weighed future utility
    val expectedValueOfSuccess = successProbability * (successMdpState.utility + mdp.getReward(action))
    val expectedValueOfFailure = failProbability * failMdpState.utility
    return expectedValueOfSuccess + expectedValueOfFailure
}

fun selectBestAction(state: BeliefState, mdp: MDP): Pair<Int, Double> {
    var bestAction: Int? = null
    var bestQValue = Double.NEGATIVE_INFINITY

    mdp.actions.forEach {
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



