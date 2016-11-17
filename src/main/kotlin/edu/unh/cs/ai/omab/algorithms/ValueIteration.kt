package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.*
import java.util.stream.IntStream

/** the algorithm the outside calls do perform value iteration*/
fun valueIteration(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
    mdp.generateStates(horizon)
    val states: MutableMap<BeliefState, BeliefState> = mdp.states

    var numberOfStates: Int = 0

    val mdpStates: MutableList<MutableList<BeliefState>> = ArrayList()
    var preLevel: MutableList<BeliefState> = ArrayList()
    var curLevel: MutableList<BeliefState> = ArrayList()

    preLevel.add(mdp.startState)
    mdpStates.add(preLevel)

    for (currentIndexLevel in 0..horizon-1) {
        curLevel = ArrayList()

        for (previousIndexLevel in 0..(preLevel.size - 1)) {
            for (actionIndex in 0..(Action.Companion.getActions().size - 1)) {
                for (successorIndex in 0..1) {
                    if (successorIndex % 2 == 0) {
                        val tempState: BeliefState = preLevel[previousIndexLevel].nextState(
                                if (actionIndex == 0) Action.LEFT else Action.RIGHT,
                                if (successorIndex == 0) true else false)
                        curLevel.add(tempState)
                        numberOfStates++
                    } else {
                        val tempState: BeliefState = preLevel[previousIndexLevel].nextState(
                                if (actionIndex == 0) Action.LEFT else Action.RIGHT,
                                if (successorIndex == 0) true else false)
                        curLevel.add(tempState)
                        numberOfStates++
                    }
                }
            }
        }
        mdpStates.add(curLevel)
        preLevel = ArrayList()
        for(state in 0..(curLevel.size-1)) {
            preLevel.add(curLevel[state])
        }
    }

    val valueIteration = ValueIteration()
    val policy = valueIteration.doValueIteration(mdpStates, numberOfStates, states)
    var currentState = mdp.startState
    return IntStream.range(0, horizon).mapToDouble {
        // Select action based on the policy
        val bestAction = policy[currentState]!!

        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState

        reward.toDouble()
    }.sum()
}

class ValueIteration() {
    fun doValueIteration(mdpStates: MutableList<MutableList<BeliefState>>, numberOfStates: Int,
                         states: MutableMap<BeliefState, BeliefState>): MutableMap<BeliefState, Action> {

        val values: MutableList<Double> = ArrayList(numberOfStates)

        val policy: MutableMap<BeliefState, Action> = HashMap()

        for (level in (mdpStates.size)..0) {
            for (stateIndex in 0..(mdpStates[level].size-1)) {
                values[stateIndex] = maxActionSum(mdpStates[level][stateIndex])
            }
        }

        calculatePolicy(policy, values, states, mdpStates)
        return policy
    }

    /** calculates optimal policy given the converged values*/
    fun calculatePolicy(policy: MutableMap<BeliefState, Action>, values: MutableList<Double>,
                        states: MutableMap<BeliefState, BeliefState>,
                        mdpStates: MutableList<MutableList<BeliefState>>) {
        for (level in 0..(mdpStates.size - 1)) {
            for (stateIndex in 0..(mdpStates[level].size - 1)) {
                val stateOptimalValue: MutableList<Double> = optimalMaxSum(states[mdpStates[level][stateIndex]]!!, values)
                val optimalAction: Int = optimalMaxAction(stateOptimalValue)
                policy[mdpStates[level][stateIndex]] = if (optimalAction == 0) Action.LEFT else Action.RIGHT
            }
        }
    }

    /** finds the optimal action given the optimal values*/
    fun optimalMaxAction(stateOptimalValue: MutableList<Double>): Int {
        var currentMax: Double = 0.0
        var currentMaxIndex: Int = -1
        if (stateOptimalValue.size > 0) {
            currentMax = stateOptimalValue[0]
            currentMaxIndex = 0
        }
        for (actionIndex in 0..(stateOptimalValue.size - 1)) {
            if (currentMax < stateOptimalValue[actionIndex]) {
                currentMax = stateOptimalValue[actionIndex]
                currentMaxIndex = actionIndex
            }
        }
        return currentMaxIndex
    }

    /** calculate the vector of transition times optimal values only called after utilities have been updated*/
    fun optimalMaxSum(state: BeliefState, values: MutableList<Double>): MutableList<Double> {
        val actionSum: MutableList<Double> = ArrayList(values.size)
        for(index in 0..(Action.Companion.getActions().size-1)) {
            actionSum.add(0.0)
        }

        for (actionIndex in 0..(Action.Companion.getActions().size - 1)) {
            for (successorIndex in 0..1) {
                if (successorIndex % 2 == 0) {
                    val successorValue: Double = state.nextState(if (actionIndex == 0) Action.LEFT else Action.RIGHT,
                            if (successorIndex == 0) true else false).utility
                    actionSum[actionIndex] += state.leftMean() * (successorValue + (if (successorIndex == 0) 1 else 0))
                } else {
                    val successorValue: Double = state.nextState(if (actionIndex == 0) Action.LEFT else Action.RIGHT,
                            if (successorIndex == 0) true else false).utility
                    actionSum[actionIndex] += state.rightMean() * (successorValue + (if (successorIndex == 0) 1 else 0))
                }
            }
        }
        return actionSum
    }

    /** takes a state, values, and state under evaluation
     * gives back the max of the sum of the state probabilities times value
     * called only while updating utilities*/
    fun maxActionSum(state: BeliefState): Double {
        val actionSum: MutableList<Double> = ArrayList(Action.Companion.getActions().size)

        for (actionIndex in 0..(Action.Companion.getActions().size - 1)) {
            for (successorIndex in 0..1) {
                if (successorIndex % 2 == 0) {
                    val successorValue: Double = state.nextState(if (actionIndex == 0) Action.LEFT else Action.RIGHT,
                            if (successorIndex == 0) true else false).utility
                    actionSum[actionIndex] += state.leftMean() * (successorValue + (if (successorIndex == 0) 1 else 0))
                } else {
                    val successorValue: Double = state.nextState(if (actionIndex == 0) Action.LEFT else Action.RIGHT,
                            if (successorIndex == 0) true else false).utility
                    actionSum[actionIndex] += state.rightMean() * (successorValue + (if (successorIndex == 0) 1 else 0))
                }

            }
        }
        return actionMax(actionSum)
    }

    /** finds the max action of a given list of action sums*/
    fun actionMax(actionSum: MutableList<Double>): Double {
        var currentMax: Double = 0.0
        if (actionSum.size > 0) {
            currentMax = actionSum[0]
        }
        (0..actionSum.size - 1)
                .asSequence()
                .filter { currentMax < actionSum[it] }
                .forEach { currentMax = actionSum[it] }
        return currentMax
    }
}


















