package edu.unh.cs.ai.omab.domain

import edu.unh.cs.ai.omab.domain.Action.LEFT
import edu.unh.cs.ai.omab.domain.Action.RIGHT
import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */

data class BeliefState(val alphaLeft: Int, val betaLeft: Int, val alphaRight: Int, val betaRight: Int) {
    var utility = 0.0

    fun leftSum() = alphaLeft + betaLeft
    fun rightSum() = alphaRight + betaRight
    fun totalSum() = leftSum() + rightSum()
    fun leftMean() = alphaLeft.toDouble() / leftSum()
    fun rightMean() = alphaRight.toDouble() / rightSum()

    fun nextState(action: Action, success: Boolean) =
            when {
                LEFT == action && success -> BeliefState(alphaLeft + 1, betaLeft, alphaRight, betaRight)
                LEFT == action && !success -> BeliefState(alphaLeft, betaLeft + 1, alphaRight, betaRight)
                RIGHT == action && success -> BeliefState(alphaLeft, betaLeft, alphaRight + 1, betaRight)
                RIGHT == action && !success -> BeliefState(alphaLeft, betaLeft, alphaRight, betaRight + 1)
                else -> throw RuntimeException("Invalid state!")
            }

    fun actionMean(action: Action) = when (action) {
        Action.LEFT -> leftMean()
        Action.RIGHT -> rightMean()
    }

    fun actionSum(action: Action) = when (action) {
        Action.LEFT -> leftSum()
        Action.RIGHT -> rightSum()
    }
}

enum class Action {
    LEFT, RIGHT;

    companion object {
        fun getActions(): List<Action> {
            val availableActions = listOf(LEFT, RIGHT)
            return availableActions
        }

        fun getReward(action: Action): Double {
            return 1.0
        }
    }
}

data class TransitionResult(val state: BeliefState, val reward: Int)

class MDP(depth: Int? = null) {
    val states: MutableMap<BeliefState, BeliefState> = HashMap()
    private val mapsByLevel: Array<MutableMap<BeliefState, BeliefState>>
    private val statesByLevel: Array<MutableList<BeliefState>>

    val startState = BeliefState(1, 1, 1, 1)

    init {
        mapsByLevel = Array<MutableMap<BeliefState, BeliefState>>(depth?.plus(1) ?: 0, { HashMap<BeliefState, BeliefState>() })
        statesByLevel = Array<MutableList<BeliefState>>(depth?.plus(1) ?: 0, { ArrayList<BeliefState>() })
    }


    fun addStates(statesToAdd: ArrayList<BeliefState>) {
        statesToAdd.forEach {
            val level = it.alphaLeft + it.betaLeft + it.alphaRight + it.betaRight - 4
            mapsByLevel[level][it] = it
            statesByLevel[level].add(it)
            states[it] = it
        }
    }

    fun generateStates(depth: Int) {
        val sum = depth + 4 //  4 is the prior
        for (leftAlpha in 1..sum) {
            for (leftBeta in 1..(sum - leftAlpha)) {
                for (rightAlpha in 1..(sum - leftAlpha - leftBeta)) {
                    for (rightBeta in 1..(sum - leftAlpha - leftBeta - rightAlpha)) {
                        val state = BeliefState(leftAlpha, leftBeta, rightAlpha, rightBeta)
                        count++
                        states[state] = state
                        val level = leftAlpha + leftBeta + rightAlpha + rightBeta - 4 // 4 is the prior
                        mapsByLevel[level][state] = state
                        statesByLevel[level].add(state)
                    }
                }
            }
        }
    }

    fun generateStates(depth: Int, state: BeliefState): ArrayList<BeliefState> {
        val initializedStates = ArrayList<BeliefState>()
        for (x in 0..(depth)) {
            for (y in 0..(depth - x)) {
                (0..(depth - x - y))
                        .mapTo(initializedStates) {
                            BeliefState(x + state.alphaLeft, y + state.betaLeft,
                                    it + state.alphaRight, depth - x - y - it + state.betaRight)
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

abstract class Simulator() {
    val random = Random()
    fun bernoulli(probability: Double): Boolean = random.nextDouble() <= probability
    abstract fun transition(state: BeliefState, action: Action): TransitionResult
}

class BanditSimulator() : Simulator() {
    override fun transition(state: BeliefState, action: Action): TransitionResult {
        return when (action) {
            LEFT -> {
                val success = bernoulli(state.leftMean())
                TransitionResult(state.nextState(LEFT, success), if (success) 1 else 0)
            }
            RIGHT -> {
                val success = bernoulli(state.rightMean())
                TransitionResult(state.nextState(RIGHT, success), if (success) 1 else 0)
            }
        }
    }
}

class BanditWorld(val leftProbability: Double, val rightProbability: Double) : Simulator() {
    override fun transition(state: BeliefState, action: Action): TransitionResult {
        return when (action) {
            LEFT -> {
                val success = bernoulli(leftProbability)
                TransitionResult(state.nextState(LEFT, success), if (success) 1 else 0)
            }
            RIGHT -> {
                val success = bernoulli(rightProbability)
                TransitionResult(state.nextState(RIGHT, success), if (success) 1 else 0)
            }
        }
    }
}

