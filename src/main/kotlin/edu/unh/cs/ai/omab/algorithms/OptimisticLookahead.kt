package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.ConfigurationExtras.*
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.utils.betaDistributions
import edu.unh.cs.ai.omab.utils.maxValueBy
import edu.unh.cs.ai.omab.utils.pow
import org.apache.commons.math3.distribution.BetaDistribution
import java.lang.Math.min
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun executeStochasticAlgorithm(world: Simulator, simulator: Simulator, configuration: Configuration, algorithm: (BeliefState, Configuration, Random) -> Int): List<Double> {
    val mdp: MDP = MDP(numberOfActions = configuration.arms)
    mdp.setRewards(configuration.rewards)

    var currentState: BeliefState = mdp.startState
    var augmentedState: BeliefState = currentState
    val rewards: MutableList<Double> = ArrayList(configuration.horizon)
    val random = world.random

    (0..configuration.horizon - 1).forEach {
        val bestAction = algorithm(augmentedState, configuration, random)

        val (nextState, reward) = world.transition(currentState, bestAction)
//        println("current: $currentState next: $nextState action: $bestAction reward: $reward")
        currentState = nextState
        augmentedState = currentState

        rewards.add(reward)
    }

    return rewards
}

fun optimisticLookahead(state: BeliefState, configuration: Configuration, random: Random): Int {
    val lookahead = min(configuration[LOOKAHEAD] as Int, configuration.horizon)
    val remainingSteps = configuration.horizon - state.totalSteps() - lookahead
    val discountFactor = configuration[DISCOUNT] as Double
    val discountedRemainingSteps = (1 - (discountFactor pow remainingSteps)) / (1 - discountFactor)
    val betaSampleCount = configuration[BETA_SAMPLE_COUNT] as Int
    val constrainedProbabilities = configuration[CONSTRAINED_PROBABILITIES] as Boolean

    val exploredStates = hashMapOf<BeliefState, BeliefState>()

    fun lookahead(state: BeliefState, currentDepth: Int, maximumDepth: Int): Double {
        return if (currentDepth >= maximumDepth) {
            sampleBetaValue(state, betaSampleCount, random, constrainedProbabilities, configuration) * discountedRemainingSteps
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
                val utility = probability * (configuration.rewards[it.first.action] + successUtility) + (1 - probability) * failUtility
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
        val utility = probability * (configuration.rewards[it.first.action] + successUtility) + (1 - probability) * failUtility
        utility
    }!!.first.action
}

private fun sampleBetaValue(state: BeliefState, count: Int, random: Random, constraints: Boolean, configuration: Configuration): Double {
    // If probability constraints are enabled remove the inconsistent probabilities
    return if (constraints) {
        // Create a beta distribution for each arm
        val betaDistributions = state.betaDistributions()
        (1..count)
                .map { betaDistributions.map(BetaDistribution::sample) }
                .filter { list -> list.indices.drop(1).all { list[it - 1] >= list[it] } }
                .map { armProbabilities -> state.arms.maxValueBy { arm -> armProbabilities[arm] * configuration.rewards[arm] }!! }
                .average()
    } else {
        val distributions = edu.unh.cs.ai.omab.utils.UncommonDistributions()
        DoubleArray(count) {
            //            betaDistributions.maxValueBy(BetaDistribution::sample)!!
            state.arms.maxValueBy { distributions.rBeta(state.alphas[it].toDouble(), state.betas[it].toDouble()) * configuration.rewards[it] }!!
        }.average()
    }
}

fun thompsonSampling(state: BeliefState, configuration: Configuration, random: Random): Int {
    val distributions = state.betaDistributions()

    val samples = distributions.mapIndexed { arm, betaDistribution ->
        betaDistribution.sample() * configuration.rewards[arm]
    }
    return (0..samples.size - 1).maxBy { samples[it] }!!
}

fun upperConfidenceBounds(state: BeliefState, configuration: Configuration, random: Random): Int {
    return state.arms.maxBy {
        upperConfidenceBoundsValue(state.actionMean(it), state.totalSteps() + 1, state.actionSum(it) - 1, 2.0) * configuration.rewards[it]
    }!!
}

fun evaluateStochasticAlgorithm(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration, algorithm: (BeliefState, Configuration, Random) -> Int, name: String): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)

    val expectedMaxReward = (configuration.rewards zip probabilities).maxValueBy { it.first * it.second }!!

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

    val averageRegrets = averageRewards.map { reward -> expectedMaxReward - reward }
    var sum = 0.0
    val cumSumRegrets = averageRegrets.map {
        sum += it
        sum
    }

    results.add(Result("$name", probabilities, expectedMaxReward, averageRegrets.last(), expectedMaxReward - averageRegrets.last(), averageRegrets, cumSumRegrets))

    return results
}