package edu.unh.cs.ai.omab.domain

import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
abstract class Simulator() {
    val random = Random()
    fun bernoulli(probability: Double): Boolean = random.nextDouble() <= probability
    abstract fun transition(state: BeliefState, action: Action): TransitionResult
}

object BanditSimulator : Simulator() {
    override fun transition(state: BeliefState, action: Action): TransitionResult {
        return when (action) {
            Action.LEFT -> {
                val success = bernoulli(state.leftMean())
                TransitionResult(state.nextState(Action.LEFT, success), if (success) 1.0 else 0.0)
            }
            Action.RIGHT -> {
                val success = bernoulli(state.rightMean())
                TransitionResult(state.nextState(Action.RIGHT, success), if (success) 1.0 else 0.0)
            }
        }
    }
}

class BanditWorld(val leftProbability: Double, val rightProbability: Double) : Simulator() {
    override fun transition(state: BeliefState, action: Action): TransitionResult {
        return when (action) {
            Action.LEFT -> {
                val success = bernoulli(leftProbability)
                TransitionResult(state.nextState(Action.LEFT, success), if (success) leftProbability else 0.0)
            }
            Action.RIGHT -> {
                val success = bernoulli(rightProbability)
                TransitionResult(state.nextState(Action.RIGHT, success), if (success) rightProbability else 0.0)
            }
        }
    }
}

data class TransitionResult(val state: BeliefState, val reward: Double)
