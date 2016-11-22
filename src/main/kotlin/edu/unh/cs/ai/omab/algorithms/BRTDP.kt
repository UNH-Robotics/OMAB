package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.*
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import java.util.*
import java.util.stream.IntStream

/**
 * Created by reazul on 11/18/16.
 */

class Brtdp(val mdp: MDP, val simulator: Simulator, val simulationCount: Int, val horizon: Int, val alpha: Double, val T: Double, configuration: Configuration) {
    lateinit private var upperBound: MutableMap<BeliefState, Double>
    lateinit private var lowerBound: MutableMap<BeliefState, Double>
    lateinit private var graph: MutableMap<BeliefState, BeliefState>
    val numActions:Int
    val random = Random()

    init {
        numActions = configuration.arms - 1
    }

    fun initBoundsIfNeeded(state: BeliefState){
        if(!graph.containsKey(state)) {
            graph.put(state, state)
            lowerBound[state] = 0.0
            upperBound[state] = horizon * 1.0
        }
    }

    data class TransitionResult(val successorStates: ArrayList<BeliefState>, val successorValues: ArrayList<Double>)
    fun getSuccessors(state: BeliefState): TransitionResult {
        val successorValues = ArrayList<Double>(4)
        val successorStates = ArrayList<BeliefState>(4)

        for (action in 0..numActions) {
            for (isSuccess in listOf(true, false)) {
                val nextState = state.nextState(action, isSuccess)
                successorStates.add(nextState)
                val trp = if (isSuccess) state.actionMean(action) else 1 - state.actionMean(action)
                successorValues.add(trp * (upperBound[nextState]!! - lowerBound[nextState]!!))
            }
        }
        return TransitionResult(successorStates, successorValues)
    }

    fun sampleSuccessor(successorValues: ArrayList<Double>): Int {
        var sumProportion = 0.0
        val rand = random.nextDouble() * successorValues.sum() // generate random in range 0 to successorValues.sum()
        for (i in 0..successorValues.size-1) {
            if (rand < successorValues[i]) return i
            sumProportion += successorValues[i]
        }
        return random.nextInt(successorValues.size)
    }

    fun calculateQValue(state: BeliefState, action: Int, isUpper: Boolean): Double {
        val successProbability = state.actionMean(action)
        val failProbability = 1 - successProbability

        val successState = state.nextState(action, true)
        val failState = state.nextState(action, false)

        initBoundsIfNeeded(successState)
        initBoundsIfNeeded(failState)

        var Qv = 0.0
        if(isUpper) Qv = 1 + (successProbability * upperBound[successState]!!) + (failProbability * upperBound[failState]!!)
        else Qv = 1 + (successProbability * lowerBound[successState]!!) + (failProbability * lowerBound[failState]!!)

        return Qv
    }

    fun getBound(state: BeliefState, isUpper: Boolean): Double{
        var maxValue = Double.NEGATIVE_INFINITY
        (0..numActions).forEach {
            val qValue = calculateQValue(state, it, isUpper)
            if(maxValue<qValue) maxValue = qValue
        }
        return maxValue
    }

    fun runSampleTrial(startState: BeliefState, level: Int){
        var state = startState
        val stack = Stack<BeliefState>()

        for(i in 0..horizon){
            stack.push(state)
            initBoundsIfNeeded(state)
            upperBound[state] = getBound(state, true)
            lowerBound[state] = getBound(state, false)

            val (successorStates, successorValues) = getSuccessors(state)
            val sumSuccessorValues = successorValues.sum()

            if (sumSuccessorValues < ((upperBound[startState]!! - lowerBound[startState]!!) / T)) break

            for (j in 0..successorValues.size-1)
                successorValues[j] = successorValues[j] / sumSuccessorValues
            state = successorStates[sampleSuccessor(successorValues)]
        }

        while(!stack.isEmpty()){
            state  = stack.pop()
            upperBound[state] = getBound(state, true)
            lowerBound[state] = getBound(state, false)
        }
    }

    fun simulate(startState: BeliefState, level: Int) {
        upperBound = HashMap()
        lowerBound = HashMap()
        graph = HashMap()
        initBoundsIfNeeded(startState)

        println("Init ConfidenceDifference: ${upperBound[startState]!! - lowerBound[startState]!!}")

        while (upperBound[startState]!! - lowerBound[startState]!! > alpha) {
            runSampleTrial(startState, level)
            println("up: ${upperBound[startState]!!}, low: ${lowerBound[startState]!!}")
        }
    }
}

fun brtdp(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator, rollOutCount: Int, configuration: Configuration): List<Double> {

    var currentState: BeliefState = mdp.startState
    val averageRewards: MutableList<Double> = ArrayList(horizon)
    val simulationCount = 200
    val eps = 0.1  //Need to make sure about this value
    val T = 50.0  //May need to tune this value by trial & error
    var sum = 0.0

    val brtdp = Brtdp(mdp, simulator, simulationCount, horizon, eps, T, configuration)

    brtdp.simulate(currentState, 0)

    /*(0..horizon - 1).forEach { level ->
        brtdp.simulate(currentState, level)
        bellmanUtilityUpdate(currentState, mdp)
        val (bestAction, bestReward) = selectBestAction(currentState, mdp)
        val (nextState, reward) = world.transition(currentState, bestAction)

        //println("current state: $currentState, bestAction: $bestAction, Reward: $reward")

        currentState = nextState
        sum += reward
        averageRewards.add(sum / (level + 1.0))
    }*/

    return averageRewards //Need to make sure about the return value & need to implement the online assumption
}

fun executeBrtdp(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration): List<Result> {
    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val rollOutCounts = intArrayOf(10, 100, 500, 1000)
    val expectedMaxReward = probabilities.max()!!

    val mdp = MDP(configuration.horizon + 1, configuration.arms)

    brtdp(mdp, configuration.horizon, world, simulator, 20, configuration)

    /*rollOutCounts.forEach { rollOutCount ->
        val rewardsList = IntStream.range(0, configuration.horizon).mapToObj {
            brtdp(mdp, configuration.horizon, world, simulator, rollOutCount, configuration)
        }

        val sumOfRewards = DoubleArray(configuration.horizon)
        rewardsList.forEach { rewards ->
            (0..configuration.horizon - 1).forEach {
                sumOfRewards[it] = rewards[it] + sumOfRewards[it]
            }
        }

        val averageRewards = sumOfRewards.map { expectedMaxReward - it / configuration.horizon }

        println("BRTDP: $rollOutCount, probabilities: $probabilities, expectedMaxReward: $expectedMaxReward, " +
                "averageRewards.last(): ${averageRewards.last()}, averageRewards: $averageRewards")

        results.add(Result("BRTDP: $rollOutCount", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))
    }*/

    return results
}