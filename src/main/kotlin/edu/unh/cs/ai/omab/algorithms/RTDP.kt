package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.*
import edu.unh.cs.ai.omab.experiment.Result
import java.util.*
import java.util.stream.IntStream

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Rtdp(val mdp: MDP, val simulator: Simulator, val simulationCount: Int, val horizon: Int) {
    private var graph: MutableMap<BeliefState, BeliefState> = HashMap()

    fun rollOut(currentState: BeliefState, currentDepth: Int) {
        if (currentDepth >= horizon) return

        if (!graph.containsKey(currentState)) graph.put(currentState, currentState)
        val statesToAdd = mdp.generateStates(1, currentState)
        mdp.addStates(statesToAdd)
        val stack = Stack<BeliefState>()

        (currentDepth..horizon).forEach {
            val (bestAction, qValue) = selectBestAction(currentState, mdp)
            val (nextState, reward) = simulator.transition(currentState, bestAction)
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

fun rtdp(horizon: Int, world: Simulator, simulator: Simulator, rollOutCount: Int): List<Double> {
    val mdp = MDP(horizon + 1)
    val rtdp = Rtdp(mdp, simulator, rollOutCount, horizon)


    var averageRewards: MutableList<Double> = ArrayList(horizon)
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

fun executeRtdp(horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int): List<Result> {
    val results: MutableList<Result> = ArrayList(iterations)
    val rollOutCounts = intArrayOf(10, 50, 100, 200, 500)
    val expectedMaxReward = probabilities.max()!!

    rollOutCounts.forEach { rollOutCount ->
        val rewardsList = IntStream.range(0, iterations).mapToObj {
            rtdp(horizon, world, simulator, rollOutCount)
        }

        val sumOfRewards = DoubleArray(horizon)
        rewardsList.forEach { rewards ->
            (0..horizon - 1).forEach {
                sumOfRewards[it] = rewards[it] + sumOfRewards[it]
            }
        }

        val averageRewards = sumOfRewards.map { expectedMaxReward - it / iterations }

        results.add(Result("rtdp$rollOutCount", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))
    }

    return results
}