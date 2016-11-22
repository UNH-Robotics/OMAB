package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.utils.sampleCorrection
import org.apache.commons.math3.distribution.BetaDistribution
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun thompsonSampling(horizon: Int, world: Simulator, arms: Int, rewards: DoubleArray, specialSauce: Boolean): List<Double> {
    val mdp: MDP = MDP(numberOfActions = arms)
    var currentState: BeliefState = mdp.startState
    mdp.setRewards(rewards)
    val rewards: MutableList<Double> = ArrayList(horizon)
    var sum = 0.0

    (0..horizon - 1).forEach { level ->

        val distributions = (0..currentState.alphas.size - 1).map {
            BetaDistribution(currentState.alphas[it].toDouble(), currentState.betas[it].toDouble())
        }

        val samples = (0..distributions.size - 1).map {
            distributions[it].inverseCumulativeProbability(world.random.nextDouble())
        }

//        val leftBetaDistribution = BetaDistribution(currentState.alphaLeft.toDouble(), currentState.betaLeft.toDouble())
//        val rightBetaDistribution = BetaDistribution(currentState.alphaRight.toDouble(), currentState.betaRight.toDouble())

//        val leftSample = leftBetaDistribution.inverseCumulativeProbability(world.random.nextDouble())
//        val rightSample = rightBetaDistribution.inverseCumulativeProbability(world.random.nextDouble())



        var bestAction = 0
        (0..samples.size - 1).forEach {
            if (samples[bestAction] < samples[it]) {
                bestAction = it
            } else {
                bestAction = bestAction
            }
        }

        if(specialSauce) {
            val newTransitions = sampleCorrection(currentState)
            if(!newTransitions[0].isNaN()) {
                world.updateTransitionProbabilities(sampleCorrection(currentState))
            }
        }

        val (nextState, reward) = world.transition(currentState, bestAction)
//        val (nextState, reward) = if (leftSample > rightSample) {
//            world.transition(currentState, LEFT)
//        } else {
//            world.transition(currentState, RIGHT)
//        }

        currentState = nextState
        sum += reward
        rewards.add(sum / (level + 1.0))
    }

    return rewards
}

fun executeThompsonSampling(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val expectedMaxReward = probabilities.max()!!

    val rewardsList = IntStream.range(0, configuration.iterations).mapToObj {
        thompsonSampling(configuration.horizon, world, configuration.arms, configuration.rewards, configuration.specialSauce)
    }

    val sumOfRewards = DoubleArray(configuration.horizon)
    rewardsList.forEach { rewards ->
        (0..configuration.horizon - 1).forEach {
            sumOfRewards[it] = rewards[it] + sumOfRewards[it]
        }
    }

    val averageRewards = sumOfRewards.map { (expectedMaxReward) - it / configuration.iterations }
    var sauceFlag = ""
    if (configuration.specialSauce) sauceFlag = "SS" else sauceFlag = sauceFlag
    results.add(Result("TS $sauceFlag", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))

    return results
}