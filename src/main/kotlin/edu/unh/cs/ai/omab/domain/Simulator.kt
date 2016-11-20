package edu.unh.cs.ai.omab.domain

import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
abstract class Simulator() {
    val random = Random()
    fun bernoulli(probability: Double): Boolean = random.nextDouble() <= probability
    abstract fun transition(state: BeliefState, action: Int): TransitionResult
}

object BanditSimulator : Simulator() {
    override fun transition(state: BeliefState, action: Int): TransitionResult {
        val success = bernoulli(state.actionMean(action))
        return TransitionResult(state.nextState(action, success), if (success) 1.0 else 0.0)
    }
}

class BanditWorld(val probabilities: DoubleArray) : Simulator() {
    override fun transition(state: BeliefState, action: Int): TransitionResult {
        val success = probabilities.map { bernoulli(it) }
        return TransitionResult(state.nextState(action, success[action]), if (success[action]) probabilities[action] else 0.0)
//                TransitionResult (state.nextState(
//                return when (action) {
//                        val success = bernoulli(leftProbability)
//                    TransitionResult(state.nextState(Action.LEFT, success), if (success) leftProbability else 0.0)
//                }
//                val success = bernoulli (rightProbability)
//                TransitionResult (state.nextState(Action.RIGHT, success), if (success) rightProbability else 0.0)
    }
}

data class TransitionResult(val state: BeliefState, val reward: Double)
