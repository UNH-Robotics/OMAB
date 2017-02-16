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
    // Acquire parameters
    val lookahead = min(configuration[LOOKAHEAD] as Int, configuration.horizon)
    val remainingSteps = configuration.horizon - state.totalSteps() - lookahead
    val discountFactor = configuration[DISCOUNT] as Double
    val betaSampleCount = configuration[BETA_SAMPLE_COUNT] as Int
    val constrainedProbabilities = configuration[CONSTRAINED_PROBABILITIES] as Boolean

    val discountedRemainingSteps = discountFactor * (1 - (discountFactor pow remainingSteps)) / (1 - discountFactor)

    val exploredStates = hashMapOf<BeliefState, BeliefState>()

    // Explore utilities using depth-first search
    fun lookahead(state: BeliefState, currentDepth: Int, maximumDepth: Int): Double {
        return if (currentDepth >= maximumDepth) {
            sampleBetaValue(state, betaSampleCount, constrainedProbabilities, configuration.rewards) * discountedRemainingSteps
        } else {
            state.successors().map {
                // Reuse the utility if already calculated
                fun calculateUtility(state: BeliefState): Double {
                    exploredStates[state] = state
                    val utility = lookahead(state, currentDepth + 1, maximumDepth)
                    state.utility = utility
                    return utility
                }

                val successState = it.first.state
                val successUtility = exploredStates[successState]?.utility ?: calculateUtility(successState)

                val failState = it.second.state
                val failUtility = exploredStates[failState]?.utility ?: calculateUtility(failState)

                val probability = state.actionMean(it.first.action)
//                val probability = sampleBetaValue(state, betaSampleCount, constrainedProbabilities, configuration.rewards)


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

/**
 * Calculate the expected utility of a state by sampling from the beta distribution of each arm.
 *
 * @param state belief state.
 * @param betaSampleCount number of beta samples.
 * @param constraints if true, filter inconsistent samples/possible probabilities.
 * @param rewards ordered arm rewards
 *
 * @return Utility of the state.
 */
private fun sampleBetaValue(state: BeliefState, betaSampleCount: Int, constraints: Boolean, rewards: DoubleArray): Double {
    val betaDistributions = state.betaDistributions()

    return if (constraints) {
        // If probability constraints are enabled remove the inconsistent probabilities
        (1..betaSampleCount)
                // Sample arm probability lists
                .map { betaDistributions.map(BetaDistribution::sample) }
                // Remove those sampled probability lists that are not conform with the prior constraints
                .filter { list -> list.indices.drop(1).all { list[it - 1] >= list[it] } }
                // Calculate the best action value for each list
                .map { armProbabilities -> state.arms.maxValueBy { arm -> armProbabilities[arm] * rewards[arm] }!! }
                .average()
    } else {
        DoubleArray(betaSampleCount) {
            state.arms.maxValueBy { arm -> betaDistributions[arm].sample() * rewards[arm] }!!
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