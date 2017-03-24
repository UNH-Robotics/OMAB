package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import org.apache.commons.math3.distribution.BetaDistribution
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun thompsonSampling(horizon: Int, world: Simulator, arms: Int, armRewards: DoubleArray, configuration: Configuration): List<Double> {
    val mdp: MDP = MDP(numberOfActions = arms)
    mdp.setRewards(armRewards)

    var currentState: BeliefState = mdp.startState
    var augmentedState: BeliefState = currentState
    val rewards: MutableList<Double> = ArrayList(horizon)

    (0..horizon - 1).forEach {
        val distributions = (0..augmentedState.alphas.size - 1).map {
            BetaDistribution(augmentedState.alphas[it].toDouble(), augmentedState.betas[it].toDouble())
        }

        val samples = distributions.map { it.inverseCumulativeProbability(world.random.nextDouble()) }
        val bestAction = (0..samples.size - 1).maxBy { samples[it] }!!

        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState


        if (!configuration.ignoreInconsistentState || currentState.isConsistent()) augmentedState = currentState
        rewards.add(reward)
    }

    return rewards
}

fun executeThompsonSampling(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val expectedMaxReward = probabilities.max()!!

    // Do multiple experiments
    val rewardsList = IntStream.range(0, configuration.iterations).mapToObj {
        thompsonSampling(configuration.horizon, world, configuration.arms, configuration.rewards, configuration)
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

//    results.add(Result("TS${if (configuration.ignoreInconsistentState) "-IGNORE" else ""}", probabilities, expectedMaxReward, averageRegrets.last(), expectedMaxReward - averageRegrets.last(), averageRegrets, cumSumRegrets))

    return results
}