package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.*
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Rtdp(val mdp: MDP, val simulator: Simulator, val simulationCount: Int, val horizon: Int) {
    private var graph: MutableMap<BeliefState, BeliefState> = HashMap()

    fun rollOut(sourceState: BeliefState, currentDepth: Int) {
        var currentState = sourceState
        if (currentDepth >= horizon) return

        if (!graph.containsKey(sourceState)) graph.put(sourceState, sourceState)
        val statesToAdd = mdp.generateNextLevel(sourceState)

        statesToAdd.forEach { mdp.addStates(mdp.generateNextLevel(it)) }

        mdp.addStates(statesToAdd)
        val stack = Stack<BeliefState>()

        (currentDepth..horizon).forEach {
            val (bestAction, qValue) = selectBestAction(sourceState, mdp)
            val (nextState, reward) = simulator.transition(sourceState, bestAction)
            currentState = nextState
            stack.push(nextState)
        }

        while (!stack.isEmpty()) {
            var curState = stack.pop()
            if (!graph.containsKey(curState)) graph.put(curState, curState)
            curState = graph[curState]!!
            mdp.addStates(mdp.generateStates(1, curState))
            bellmanUtilityUpdate(curState, mdp)
        }
    }

    fun simulate(currentState: BeliefState, currentDepth: Int) {
        (0..simulationCount).forEach { rollOut(currentState, currentDepth) }
    }
}

fun rtdp(horizon: Int, world: Simulator, simulator: Simulator, rollOutCount: Int, numberOfActions: Int): List<Double> {
    val mdp = MDP(horizon + 1, numberOfActions)
    val rtdp = Rtdp(mdp, simulator, rollOutCount, horizon)


    val averageRewards: MutableList<Double> = ArrayList(horizon)
    var sum = 0.0

    var currentState = mdp.startState
    (0..horizon - 1).forEach { level ->
        rtdp.simulate(currentState, level)
        bellmanUtilityUpdate(currentState, mdp)
        val (bestAction, bestReward) = selectBestAction(currentState, mdp)
        val (nextState, reward) = world.transition(currentState, bestAction)

        currentState = nextState
        sum += reward
        averageRewards.add(sum / (level + 1.0))
    }

    return averageRewards
}

fun executeRtdp(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val rollOutCounts = intArrayOf(10)
    val expectedMaxReward = probabilities.max()!!

    rollOutCounts.forEach { rollOutCount ->
        val rewardsList = IntStream.range(0, configuration.iterations).mapToObj {
            rtdp(configuration.horizon, world, simulator, rollOutCount, configuration.arms)
        }

        val sumOfRewards = DoubleArray(configuration.horizon)
        rewardsList.forEach { rewards ->
            (0..configuration.horizon - 1).forEach {
                sumOfRewards[it] = rewards[it] + sumOfRewards[it]
            }
        }

        val averageRewards = sumOfRewards.map { expectedMaxReward - it / configuration.iterations }

        results.add(Result("RTDP$rollOutCount", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))
    }

    return results
}