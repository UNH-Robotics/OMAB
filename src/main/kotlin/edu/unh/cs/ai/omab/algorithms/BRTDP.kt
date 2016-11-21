package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.*
import edu.unh.cs.ai.omab.experiment.Result
import java.util.*
import java.util.stream.IntStream

/**
 * Created by reazul on 11/18/16.
 */
class Brtdp(val mdp: MDP, val simulator: Simulator, val simulationCount: Int, val horizon: Int, val eps: Double, val T: Double) {
    private var graph: MutableMap<BeliefState, BeliefState> = HashMap()
    private var upperBound: MutableMap<BeliefState, Double> = HashMap()
    private var lowerBound: MutableMap<BeliefState, Double> = HashMap()
    val random = Random()

    fun getMaxQ(state: BeliefState): Double {
        return Action.getActions()
                .map { calculateQValue(checkGraph(state), it, mdp)}
                .max()!!
    }

    fun calculateQValue(st: BeliefState, action: Action, mdp: MDP): Double {
        val state = checkGraph(st)
        val successProbabily = state.actionMean(action)
        val failProbability = 1 - successProbabily

        val successState = checkGraph(state.nextState(action, true))
        val failState = checkGraph(state.nextState(action, false))

        val successorLevel = state.totalSum() - 4 + 1// 4 is the sum of priors for 2 arms
        val successMdpState = checkGraph(mdp.getLookupState(successorLevel, successState))
        val failMdpState = checkGraph(mdp.getLookupState(successorLevel, failState))

        // Calculate the probability weighed future utility
        val expectedValueOfSuccess = successProbabily * (successMdpState.utility + Action.getReward(action))
        val expectedValueOfFailure = failProbability * failMdpState.utility

        return expectedValueOfSuccess + expectedValueOfFailure
    }

    data class TransitionResult(val successorStates: ArrayList<BeliefState>, val successorValues: ArrayList<Double>)

    fun getSuccessors(state: BeliefState): TransitionResult {
        val successorValues = ArrayList<Double>(4)
        val successorStates = ArrayList<BeliefState>(4)

        for(action in Action.getActions()){
            for(isSuccess in listOf(true, false)){
                val nextState = state.nextState(action, isSuccess)
                successorStates.add(checkGraph(nextState))
                updateBounds(nextState)
                val trp = if (isSuccess) state.actionMean(action) else 1-state.actionMean(action)
                successorValues.add( trp * (upperBound[nextState]!! - lowerBound[nextState]!!) )
            }
        }
        return TransitionResult(successorStates, successorValues)
    }

    fun sampleSuccessor(successorValues: ArrayList<Double>) : Int{
        var sumProportion = 0.0
        val rand = random.nextDouble() * successorValues.sum() // generate random in range 0 to successorValues.sum()
        for(i in 0..3){
            if(rand< successorValues[i]) return i
            sumProportion += successorValues[i]
        }
        return random.nextInt(4)
    }

    fun getUpperBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
        return if (t == 1) Double.POSITIVE_INFINITY else μ + Math.sqrt(α * Math.log(t.toDouble()) / (2 * depth * (t - 1)))
    }

    fun getLowerBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
        return if (t == 1) Double.POSITIVE_INFINITY else μ - Math.sqrt(α * Math.log(t.toDouble()) / (2 * depth * (t - 1)))
    }

    fun checkGraph(state: BeliefState): BeliefState{
        if (!graph.containsKey(state)) graph.put(state, state)
        return graph[state]!!
    }

    fun updateBounds(st : BeliefState){
        val state = checkGraph(st)
        mdp.addStates(mdp.generateStates(1, state))
        val qValue = getMaxQ(state)
        var util = state.utility
        //println("Before State: $state, Utility: $util, qValue: $qValue")
        state.utility = qValue
        util = state.utility
        //println("After State: $state, Utility: $util, qValue: $qValue")
        upperBound[state] = getUpperBoundsValue(qValue, state.leftSum(), state.totalSum())
        lowerBound[state] = getLowerBoundsValue(qValue, state.leftSum(), state.totalSum())

        //println("update bounds")
        val x = lowerBound[state]!!
        val y = upperBound[state]!!
        //println("State: $state, qValue: $qValue, lower: $x, upper: $y")
    }

    fun runSampleTrial(initState: BeliefState, level: Int) : Double{
        var state = initState
        val stack = Stack<BeliefState>()

        for (i in level..horizon-1){

            state = checkGraph(state)

            stack.push(state)
            updateBounds(state)

            //println("State: $state, upperBound: ")

            val (successorStates, successorValues) = getSuccessors(state) //b(y) in paper algorithm
            val sumSuccessorValues = successorValues.sum()  //B in paper algorithm

            if(sumSuccessorValues < ((upperBound[initState]!! - lowerBound[initState]!!)/T) ) break

            for (j in 0..3) successorValues[j] = successorValues[j] / sumSuccessorValues

            //print("curState: $state, ")
            state = checkGraph(successorStates[ sampleSuccessor(successorValues) ] )
            //println("nextState: $state")
        }

        var confidenceBoundDifference = 0.0

        //println("start printing backstack")
        while(!stack.isEmpty()){
            state = stack.pop()

            state = checkGraph(state)

            updateBounds(state)
            confidenceBoundDifference = upperBound[state]!! - lowerBound[state]!!
            //println("State: $state, confidenceBoundDifference: $confidenceBoundDifference")
        }

        //println("return confidenceBoundDifference: $confidenceBoundDifference")
        return confidenceBoundDifference
    }

    fun simulate(currentState: BeliefState, level: Int) {
        var prevVal = 0.0
        var trialVal = runSampleTrial(checkGraph(currentState), level)

        while ( trialVal > eps /*&& trialVal!=prevVal*/){
            prevVal = trialVal
            trialVal = runSampleTrial(checkGraph(currentState), level)
        }
    }
}

fun brtdp(horizon: Int, world: Simulator, simulator: Simulator, rollOutCount: Int): List<Double>  {
    val mdp = MDP(horizon + 1)
    var currentState: BeliefState = mdp.startState
    val averageRewards: MutableList<Double> = ArrayList(horizon)
    val simulationCount = 200
    val eps = 0.1  //Need to make sure about this value
    val T = 50.0  //May need to tune this value by trial & error
    var sum = 0.0

    val brtdp = Brtdp(mdp, simulator, simulationCount, horizon, eps, T)
    //brtdp.simulate(currentState, 0)

    (0..horizon - 1).forEach { level ->
        brtdp.simulate(currentState, level)
        bellmanUtilityUpdate(currentState, mdp)
        val (bestAction, bestReward) = selectBestAction(currentState, mdp)
        val (nextState, reward) = world.transition(currentState, bestAction)

        //println("current state: $currentState, bestAction: $bestAction, Reward: $reward")

        currentState = nextState
        sum += reward
        averageRewards.add(sum / (level + 1.0))
    }

    return averageRewards //Need to make sure about the return value & need to implement the online assumption
}

fun executeBrtdp(horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int): List<Result> {
    val results: MutableList<Result> = ArrayList(iterations)
    val rollOutCounts = intArrayOf(10, 100, 500, 1000)
    val expectedMaxReward = probabilities.max()!!

    brtdp(horizon, world, simulator, 20)

    rollOutCounts.forEach { rollOutCount ->
        val rewardsList = IntStream.range(0, iterations).mapToObj {
            brtdp(horizon, world, simulator, rollOutCount)
        }

        val sumOfRewards = DoubleArray(horizon)
        rewardsList.forEach { rewards ->
            (0..horizon - 1).forEach {
                sumOfRewards[it] = rewards[it] + sumOfRewards[it]
            }
        }

        val averageRewards = sumOfRewards.map { expectedMaxReward - it / iterations }

        println("BRTDP: $rollOutCount, probabilities: $probabilities, expectedMaxReward: $expectedMaxReward, " +
                "averageRewards.last(): ${averageRewards.last()}, averageRewards: $averageRewards")

        results.add(Result("BRTDP: $rollOutCount", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))
    }

    return results
}