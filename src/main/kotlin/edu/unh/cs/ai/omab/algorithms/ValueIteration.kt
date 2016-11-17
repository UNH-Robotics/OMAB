package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.*

//data class Successor(val state: Int, val probability: Double)
//
//data class Action(val numberOfSuccessors: Int) {
//
//    val successors: MutableList<Successor> = ArrayList()
//
//    fun addSuccessor(successor: Successor) {
//        successors.add(successor)
//    }
//
//}
//
//data class State(val reward: Double, val isTerminal: Int, val numberOfActions: Int) {
//
//    val actions: MutableList<Action> = ArrayList()
//
//    fun addAction(action: Action) {
//        actions.add(action)
//    }
//
//}

/** the algorithm the outside calls do perform value iteration*/
fun valueIteration(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Long {
    val states: MutableMap<BeliefState, BeliefState> = mdp.states

    val valueIteration = ValueIteration(mdp, horizon, world, simulator)
    TODO()
}

class ValueIteration(val mdp: MDP, val horizon: Int, val world: Simulator, val simulator: Simulator) {
    /** the myth the legend value iteration~~~~*/
    fun doValueIteration(mdpStates: MutableList<MutableList<BeliefState>>, numberOfStates: Int,
                         states: MutableMap<BeliefState, BeliefState>): MutableMap<BeliefState, Action> {

        val values: MutableList<Double> = ArrayList(numberOfStates)
        val valuesPrime: MutableList<Double> = ArrayList(numberOfStates)

        val policy: MutableMap<BeliefState, Action> = HashMap()

        for (stateIndex in 0..(states.size - 1)) {
            printValues(values)
            valuesPrime[stateIndex] = maxActionSum(TODO())
        }

        calculatePolicy(policy, values, states)
        return policy
    }

    /** debug functions for printing*/
    fun printValues(values: MutableList<Double>) {
        values.forEach(::println)
    }

    /** calculates optimal policy given the converged values*/
    fun calculatePolicy(policy: MutableMap<BeliefState, Action>, values: MutableList<Double>, states: MutableMap<BeliefState, BeliefState>) {
        for (stateIndex in 0..(states.size - 1)) {
            val stateOptimalValue: MutableList<Double> = optimalMaxSum(states[])
            val optimalAction: Int = optimalMaxAction(stateOptimalValue)
            policy[stateIndex] = optimalAction
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
    fun optimalMaxSum(state: BeliefState, values: MutableList<Double>, stateIndex: Int): MutableList<Double> {
        val actionSum: MutableList<Double> = ArrayList(values.size)

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


















