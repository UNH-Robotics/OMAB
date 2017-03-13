package edu.unh.cs.ai.omab.domain

import edu.unh.cs.ai.omab.utils.maxIndexAfter
import edu.unh.cs.ai.omab.utils.minIndexBefore
import edu.unh.cs.ai.omab.utils.smallerIndicesBefore
import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
data class BeliefState(val alphas: IntArray, val betas: IntArray) {
    var utility = 1000.0

    fun alphaSum() = alphas.sum()
    fun betaSum() = betas.sum()
    fun totalSum() = alphaSum() + betaSum()
    fun totalSteps() = totalSum() - size * 2

    val size: Int
        get() = alphas.size

    val arms
        get() = alphas.indices

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

    override fun toString(): String {
        val builder = StringBuilder("State: ")
        (alphas zip betas).forEach { builder.append("[${it.first} ${it.second}]") }
        return builder.toString()
    }

    fun nextState(action: Int, success: Boolean): BeliefState {
        assert(action >= 0 && action < alphas.size - 1)
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

    /**
     * @return True if all arm's probability is higher than the arm's probability on its right, else false.
     */
    fun isConsistent(): Boolean = alphas.indices.drop(1).all { actionMean(it - 1) >= actionMean(it) }

    /**
     * Create a new BeliefState that is consistent with the prior knowledge.
     */
    fun augment(): BeliefState {
//        return this
        val augmentedAlphas = alphas.copyOf()
        val augmentedBetas = betas.copyOf()

        val probabilities = DoubleArray(size, { actionMean(it) })

        // Skip augmentation if the state is already consistent
        val consistent = probabilities.indices.drop(1).all { probabilities[it - 1] >= probabilities[it] }
        if (consistent) return this

        val originalProbabilities = probabilities.copyOf()
        val safe = BooleanArray(size, { false })

        fun balance(indices: List<Int>) {
            // Always use the original probabilities and counts for balancing

            val totalWeight = indices.sumBy { actionSum(it) }
            val weightedProbabilitySum = indices.sumByDouble { originalProbabilities[it] * actionSum(it) }

            val weightedProbabilityAverage = weightedProbabilitySum / totalWeight

            // Hacky approximation (additive)
            indices.forEach {
                val probability = originalProbabilities[it]
                if (probability == weightedProbabilityAverage) return@forEach

                if (probability > weightedProbabilityAverage) {
                    // Probability has to be decreased by increasing beta
                    val α = alphas[it]
                    val β = betas[it]

                    val x = α / weightedProbabilityAverage - (α + β)

                    augmentedAlphas[it] = α
                    augmentedBetas[it] = β + Math.round(x).toInt()

                } else {
                    // Probability has to be increased by increasing alpha
                    val α = alphas[it]
                    val β = betas[it]


                    val x = (weightedProbabilityAverage * (α + β) - α) / (1 + weightedProbabilityAverage)

                    augmentedAlphas[it] = α + Math.round(x).toInt()
                    augmentedBetas[it] = β
                }

                probabilities[it] = augmentedAlphas[it].toDouble() / (augmentedAlphas[it] + augmentedBetas[it])
            }
        }

        // We don't have to check the last state
        alphas.indices.forEach {
            when {
            // We are safe
                probabilities.maxIndexAfter(it) == it && probabilities.minIndexBefore(it) == it -> safe[it]// This is safe
            // We are not safe but should not do anything
                probabilities.maxIndexAfter(it) != it -> Unit
            // We are not safe and we should fix things
                else -> {
                    val indicesToBalance = probabilities.smallerIndicesBefore(it)!!
                    indicesToBalance.add(it)
                    balance(indicesToBalance)
                }

            // We should fix everything on the right or wait to be fixed
            // If we fixed on the right we can check for safety again
            // If we are safe after fix all nodes we fixed including us are safe
            }

//            val lowerAlpha = alphas[it]
//            val lowerBeta = betas[it]
        }

        val augmentedState = BeliefState(augmentedAlphas, augmentedBetas)
        System.out.println("\nOriginal: $this")
        System.out.println("Augmented: $augmentedState")

        return augmentedState
    }

    fun successors(): List<Pair<SuccessorBundle, SuccessorBundle>> = alphas.indices
            .map {
                SuccessorBundle(nextState(it, true), true, it) to SuccessorBundle(nextState(it, false), false, it)
            }

    data class SuccessorBundle(val state: BeliefState, val success: Boolean, val action: Int)

//    fun recursiveAugment(): BeliefState {
//        if (isConsistent()) return this
//
//        val queue = ArrayDeque<BeliefState>()
//        queue += this.successors()
//
//        while (queue.isNotEmpty()) {
//            val state = queue.removeFirst()
//
//            if (state.isConsistent()) {
////                System.out.println("\nOriginal: $this")
////                System.out.println("Augmented: $state")
//                return state
//            }
//            queue += state.successors()
//        }
//
//        throw RuntimeException("Consistent state is not reachable")
//    }

}

class MDP(depth: Int? = null, val numberOfActions: Int) {

    val states: MutableMap<BeliefState, BeliefState> = HashMap()
    private val mapsByLevel = Array<MutableMap<BeliefState, BeliefState>>(depth?.plus(1) ?: 0, { HashMap<BeliefState, BeliefState>() })
    private val statesByLevel = Array<MutableList<BeliefState>>(depth?.plus(1) ?: 0, { ArrayList<BeliefState>() })
    private val rewards = doubleArrayOf(1.0, 1.0, 1.0)

    val startState = BeliefState(IntArray(numberOfActions, { 1 }), IntArray(numberOfActions, { 1 }))
    val actions = IntArray(numberOfActions, { it })

    fun getReward(action: Int): Double {
        return rewards[action]
    }

    fun setRewards(newRewards: DoubleArray) {
        assert(newRewards.size == rewards.size)
        (0..newRewards.size - 1).forEach { rewards[it] = newRewards[it] }
    }

    fun addStates(statesToAdd: Iterable<BeliefState>) {
        statesToAdd.forEach {
            val level = it.alphaSum() + it.betaSum() - (2 * numberOfActions)
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
            return generateStates(0, depth, state)
        }
    }

    fun generateNextLevel(state: BeliefState): ArrayList<BeliefState> {
        val nextLevel = ArrayList<BeliefState>()
        (0..numberOfActions - 1).forEach {
            val action = it
            listOf(true, false).forEach {
                nextLevel.add(state.nextState(action, it))
            }
        }
        return nextLevel
    }

    private fun makeUnique(states: ArrayList<BeliefState>): ArrayList<BeliefState> {
        val uniqueStates = ArrayList<BeliefState>()
        states.forEach {
            if (!uniqueStates.contains(it)) uniqueStates.add(it) else {
            }
        }

        return uniqueStates
    }

    private fun generateStates(depth: Int, level: Int, state: BeliefState): ArrayList<BeliefState> {
        var levelGeneration = ArrayList<BeliefState>()
        val currentLevel = ArrayList<BeliefState>()
        currentLevel.add(state)
        if (level == 0) {
            return currentLevel
        }

//        println(state)
        (0..level - 1).forEach {
            (0..currentLevel.size - 1).forEach {
                val currentState = currentLevel[it]
                (0..numberOfActions - 1).forEach { action ->
                    val levelReturn = ArrayList<BeliefState>()
                    listOf(true, false).forEach { success ->
                        val newState = currentState.nextState(action, success)
                        val swapState = BeliefState(newState.betas, newState.alphas)
                        currentLevel.add(newState)
                        currentLevel.add(swapState)
                        currentLevel.remove(currentState)
                        levelGeneration.add(newState)
                        levelGeneration.add(swapState)
                    }
                    levelReturn.forEach { levelGeneration.add(it) }
                    levelGeneration = makeUnique(levelGeneration)
                }
            }
        }
//        return levelGeneration
        return ArrayList(levelGeneration.filter { it.totalSum() - (2 * numberOfActions) == level })
    }

//    fun generateStates(state: BeliefState, depth: Int) {
//        val queue = ArrayList<BeliefState>()
//        val set = hashSetOf<BeliefState>()
//
//        (1..depth).forEach {
//            queue.forEach {
//                set += it.successors()
//                addStates(set)
//            }
//        }
//
//    }

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



