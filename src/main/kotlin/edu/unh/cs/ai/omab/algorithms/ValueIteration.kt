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
                         states: MutableMap<BeliefState, BeliefState>): MutableList<Int> {

        val values: MutableList<Double> = ArrayList(numberOfStates)
        val valuesPrime: MutableList<Double> = ArrayList(numberOfStates)

        var terminate: Boolean = false

        var backups: Int = 0
        val policy: Map<BeliefState, Action> = HashMap()

        while (!terminate) {
            for (stateIndex in 0..(states.size - 1)) {
                printValues(values)
                valuesPrime[stateIndex] = maxActionSum(states[stateIndex], values, stateIndex)
                backups += 1
            }

            if (backups == states.size - 1) {
                terminate = true
            }

        }
        calculatePolicy(policy, values, states)
        return policy
    }

    /** debug functions for printing*/
    fun printValues(values: MutableList<Double>) {
        values.forEach(::println)
    }

    /** calculates optimal policy given the converged values*/
    fun calculatePolicy(policy: MutableList<Int>, values: MutableList<Double>, states: MutableList<State>) {
        for (stateIndex in 0..(states.size - 1)) {
            val stateOptimalValue: MutableList<Double> = optimalMaxSum(states[stateIndex], values, stateIndex)
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

    /** calculate the vector of transition times optimal values*/
    fun optimalMaxSum(state: State, values: MutableList<Double>, stateIndex: Int): MutableList<Double> {
        val actionSum: MutableList<Double> = ArrayList(values.size)

        for (actionIndex in 0..(state.actions.size - 1)) {
            for (successorIndex in 0..(state.actions[actionIndex].successors.size)) {
                actionSum += state.actions[actionIndex].successors[successorIndex].probability *
                        values[state.actions[actionIndex].successors[successorIndex].state]
            }
        }
        return actionSum
    }

    /** takes a state, values, and state under evaluation
     * gives back the max of the sum of the state probabilities times value*/
    fun maxActionSum(state: State, values: MutableList<Double>, stateIndex: Int): Double {
        val actionSum: MutableList<Double> = ArrayList(state.numberOfActions)

        for (actionIndex in 0..(state.actions.size - 1)) {
            for (successorIndex in 0..(state.actions[actionIndex].successors.size - 1)) {
                actionSum[actionIndex] += state.actions[actionIndex].successors[successorIndex].probability *
                        values[state.actions[actionIndex].successors[successorIndex].state]
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


















