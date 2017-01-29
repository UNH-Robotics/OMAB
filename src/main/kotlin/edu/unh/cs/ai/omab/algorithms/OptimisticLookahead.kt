package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.utils.betaDistributions
import edu.unh.cs.ai.omab.utils.maxValueBy
import edu.unh.cs.ai.omab.utils.pow
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun executeStochasticAlgorithm(world: Simulator, simulator: Simulator, configuration: Configuration, algorithm: (BeliefState, Int, Random) -> Int): List<Double> {
    val mdp: MDP = MDP(numberOfActions = configuration.arms)
    mdp.setRewards(configuration.rewards)

    var currentState: BeliefState = mdp.startState
    var augmentedState: BeliefState = currentState
    val rewards: MutableList<Double> = ArrayList(configuration.horizon)
    val random = world.random

    (0..configuration.horizon - 1).forEach {
        val bestAction = algorithm(augmentedState, configuration.horizon, random)

        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState
        augmentedState = currentState

        rewards.add(reward)
    }

    return rewards
}

private fun optimisticLookahead(state: BeliefState, horizon: Int, random: Random): Int {
    val remainingSteps = horizon - state.totalSteps()
    val discountFactor = 0.95
    val discountedRemainingSteps = (1 - (discountFactor pow remainingSteps)) / (1 - discountFactor)
    val betaSampleCount = 100
    val lookahead = 4

    val exploredStates = hashMapOf<BeliefState, BeliefState>()

    fun lookahead(state: BeliefState, currentDepth: Int, maximumDepth: Int): Double {
        return if (currentDepth >= maximumDepth) {
            sampleBetaValue(state, betaSampleCount, random, false)
        } else {

            state.successors().map {
                // Reuse the utility if already calculated
                val successUtility = exploredStates[it.first.state]?.utility ?: {
                    exploredStates[it.first.state] = it.first.state
                    val utility = lookahead(it.first.state, currentDepth + 1, maximumDepth)
                    it.first.state.utility = utility
                    utility
                }()

                val failUtility = exploredStates[it.second.state]?.utility ?: {
                    exploredStates[it.second.state] = it.second.state
                    val utility = lookahead(it.second.state, currentDepth + 1, maximumDepth)
                    it.second.state.utility = utility
                    utility
                }()

                val probability = state.actionMean(it.first.action)
                // Multiply the utility by the probability of getting to the state
                val utility =  probability * (1 /** Add real reward */ + successUtility) + (1 - probability) * failUtility
                utility
            }.max()!!

            // The value of a state equals to its best arm (Q)
        }
    }

    // Populate the utilities
    lookahead(state, 0, lookahead)

    // Pick the arm with the best Q
    return state.successors().maxBy {
        val successUtility = exploredStates[it.first.state]!!.utility
        val failUtility = exploredStates[it.second.state]!!.utility
        val probability = state.actionMean(it.first.action)
        val utility =  probability * (1 /** Add real reward */ + successUtility) + (1 - probability) * failUtility
        utility
    }!!.first.action
}

private fun sampleBetaValue(state: BeliefState, count: Int, random: Random, constraints: Boolean): Double {
    val betaDistributions = state.betaDistributions()

    return DoubleArray(count) {
        betaDistributions.maxValueBy {
            it.inverseCumulativeProbability(random.nextDouble())
        }!!
    }.average()
}

private fun thompshonSampling(state: BeliefState, horizon: Int, random: Random): Int {
    val distributions = state.betaDistributions()

    val samples = distributions.map { it.inverseCumulativeProbability(random.nextDouble()) }
    return (0..samples.size - 1).maxBy { samples[it] }!!
}

private fun upperConfidenceBounds(state: BeliefState, horizon: Int, random: Random): Int {
    val upperConfidenceBoundsValues = (0..state.alphas.size - 1).map {
        upperConfidenceBoundsValue(state.actionMean(it), state.actionSum(it), state.totalSum(), 2.0)
    }.toDoubleArray()

    return (0..upperConfidenceBoundsValues.size - 1).maxBy { upperConfidenceBoundsValues[it] }!!
}

fun evaluateStochasticAlgorithm(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration, algorithm: (BeliefState, Int, Random) -> Int, name: String): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val expectedMaxReward = probabilities.max()!!

    // Do multiple experiments
    val rewardsList = IntStream.range(0, configuration.iterations).mapToObj {
        executeStochasticAlgorithm(world, simulator, configuration, algorithm)
    }

    val averageRewards = DoubleArray(configuration.horizon)
    rewardsList.forEach { rewards ->
        (0..configuration.horizon - 1).forEach {
            averageRewards[it] = rewards[it] / configuration.iterations + averageRewards[it]
        }
    }

    val averageRegrets = averageRewards.mapIndexed { level, reward -> expectedMaxReward - reward }
    var sum = 0.0
    val cumSumRegrets = averageRegrets.map {
        sum += it
        sum
    }

    results.add(Result("$name", probabilities, expectedMaxReward, averageRegrets.last(), expectedMaxReward - averageRegrets.last(), averageRegrets, cumSumRegrets))

    return results
}