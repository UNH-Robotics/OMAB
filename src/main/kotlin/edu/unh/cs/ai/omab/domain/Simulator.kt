package edu.unh.cs.ai.omab.domain

import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
abstract class Simulator(val rewards: DoubleArray) {
    val random = Random()
    abstract fun updateTransitionProbabilities(newProbabilities: DoubleArray): Unit
    fun bernoulli(probability: Double): Boolean = random.nextDouble() <= probability
    abstract fun transition(state: BeliefState, action: Int): TransitionResult
}

class BanditSimulator(rewards: DoubleArray) : Simulator(rewards) {
    override fun updateTransitionProbabilities(newProbabilities: DoubleArray) {
        throw UnsupportedOperationException("not implemented")
    }

    override fun transition(state: BeliefState, action: Int): TransitionResult {
        val success = bernoulli(state.actionMean(action))
        return TransitionResult(state.nextState(action, success), if (success) rewards[action] else 0.0)
    }
}

class BanditWorld(var probabilities: DoubleArray, rewards: DoubleArray) : Simulator(rewards) {
    override fun updateTransitionProbabilities(newProbabilities: DoubleArray) {
        probabilities = newProbabilities.copyOf()
    }

    override fun transition(state: BeliefState, action: Int): TransitionResult {
        val success = bernoulli(probabilities[action])
//        return TransitionResult(state.nextState(action, success), if (success) rewards[action] else 0.0)
        // Return the expected reward except the real reward
        return TransitionResult(state.nextState(action, success), rewards[action] * probabilities[action])
    }
}

data class TransitionResult(val state: BeliefState, val reward: Double)
