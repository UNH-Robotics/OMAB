package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.*
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun onlineValueIteration(horizon: Int, world: Simulator, simulator: Simulator): Double {
    val lookAhead: Int = 10
    var realHorizon = horizon
    val onlineMDP = MDP(horizon + lookAhead, numberOfActions = 2)

    var currentState: BeliefState = onlineMDP.startState

    val addStartState = ArrayList<BeliefState>()
    addStartState.add(currentState)
    onlineMDP.addStates(addStartState)

    return IntStream.range(0, realHorizon).mapToDouble {
        (1..(lookAhead)).forEach {
            val generatedDepthStates: ArrayList<BeliefState> = onlineMDP.generateStates(it, currentState)
            generatedDepthStates.forEach { it.utility = 0.0 }

            onlineMDP.addStates(generatedDepthStates)
        }

        (lookAhead - 1 downTo 0).forEach {
            onlineMDP.getStates(it).forEach { bellmanUtilityUpdate(it, onlineMDP) }
        }
        realHorizon -= lookAhead
        val (bestAction, qValue) = selectBestAction(currentState, onlineMDP)
        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState
        reward
    }.sum()

}

fun valueIteration(mdp: MDP, horizon: Int, world: Simulator): List<Double> {
    val rewards: MutableList<Double> = ArrayList(horizon)
    var currentState = mdp.startState
    (0..horizon - 1).forEach { level ->
        // Select action based on the policy
        val (bestAction, qValue) = selectBestAction(currentState, mdp)

        val (nextState, reward) = world.transition(currentState, bestAction)
        currentState = nextState

        rewards.add(reward)
    }

    return rewards
}

fun initializeMDP(horizon: Int, arms: Int): MDP {
    val mdp = MDP(horizon, numberOfActions = arms)

    (0..horizon).forEach {
        val generatedDepthStates: ArrayList<BeliefState> = mdp.generateStates(it, mdp.startState)
        generatedDepthStates.forEach { it.utility = 0.0 }
        mdp.addStates(generatedDepthStates)
    }

    // Back up values
    (horizon - 1 downTo 0).forEach {
        mdp.getStates(it).forEach { bellmanUtilityUpdate(it, mdp) }
    }

    return mdp
}

fun executeValueIteration(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val expectedMaxReward = probabilities.max()!!

    val mdp = initializeMDP(configuration.horizon, configuration.arms)

    val rewardsList = IntStream.range(0, configuration.iterations).mapToObj {
        valueIteration(mdp, configuration.horizon, world)
    }

    val averageRewards = DoubleArray(configuration.horizon)
    rewardsList.forEach { rewards ->
        (0..configuration.horizon - 1).forEach {
            averageRewards[it] = rewards[it] + averageRewards[it]
        }
    }

    val averageRegrets = averageRewards.mapIndexed { level, reward -> expectedMaxReward - reward}
    var sum = 0.0
    val cumSumRegrets = averageRegrets.map {
        sum += it
        sum
    }
    results.add(Result("VI", probabilities, expectedMaxReward, averageRegrets.last(), expectedMaxReward - averageRegrets.last(), averageRegrets, cumSumRegrets))

    return results
}

fun calculateLookAhead(horizon: Int): HashMap<Int, Int> {

    val stateNumberToDepth = HashMap<Int, Int>()

    (0..horizon).forEach {
        val numberOfStatesGivenDepth = (6.0 * it + 11.0 * (it * it) +
                6 * (it * it * it) +
                (it * it * it * it)) / 24
        stateNumberToDepth[numberOfStatesGivenDepth.toInt()] = it
    }


    return stateNumberToDepth
}

fun calculateLookAhead(mdp: MDP, horizon: Int, world: Simulator,
                       simulator: Simulator, numberOfStates: Double): HashMap<Int, Int> {

    val stateNumberToDepth = HashMap<Int, Int>()

    (0..horizon).forEach {
        val numberOfStatesGivenDepth = (6.0 * it + 11.0 * (it * it) +
                6 * (it * it * it) +
                (it * it * it * it)) / 24
        stateNumberToDepth[numberOfStatesGivenDepth.toInt()] = it
    }


    return stateNumberToDepth
}