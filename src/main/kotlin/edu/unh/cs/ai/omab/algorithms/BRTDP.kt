//package edu.unh.cs.ai.omab.algorithms
//
//import edu.unh.cs.ai.omab.domain.*
//import edu.unh.cs.ai.omab.experiment.Result
//import java.util.*
//import java.util.stream.IntStream
//
///**
// * Created by reazul on 11/18/16.
// */
//
//class Brtdp(val mdp: MDP, val simulator: Simulator, val simulationCount: Int, val horizon: Int, val eps: Double, val T: Double) {
//    private var graph: MutableMap<BeliefState, BeliefState> = HashMap()
//    private var upperBound: MutableMap<BeliefState, Double> = HashMap()
//    private var lowerBound: MutableMap<BeliefState, Double> = HashMap()
//    val random = Random()
//
//    fun getMaxQ(state: BeliefState): Double {
//        return Action.getActions()
//                .map { calculateQ(state, it, mdp) }
//                .max()!!
//    }
//
//    data class TransitionResult(val successorStates: ArrayList<BeliefState>, val successorValues: ArrayList<Double>)
//
//    fun getSuccessors(state: BeliefState): TransitionResult {
//        val successorValues = ArrayList<Double>(4)
//        val successorStates = ArrayList<BeliefState>(4)
//
//        for(action in Action.getActions()){
//            for(isSuccess in listOf(true, false)){
//                val nextState = state.nextState(action, isSuccess)
//                successorStates.add(nextState)
//                updateBounds(nextState)
//                val trp = if (isSuccess) state.actionMean(action) else 1-state.actionMean(action)
//                successorValues.add( trp * (upperBound[nextState]!! - lowerBound[nextState]!!) )
//            }
//        }
//        return TransitionResult(successorStates, successorValues)
//    }
//
//    fun sampleSuccessor(successorValues: ArrayList<Double>) : Int{
//        var sumProportion = 0.0
//        val rand = random.nextDouble() * successorValues.sum() // generate random in range 0 to successorValues.sum()
//        for(i in 0..3){
//            if(rand< successorValues[i]) return i
//            sumProportion += successorValues[i]
//        }
//        return random.nextInt(4)
//    }
//
//    fun getUpperBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
//        return if (t == 1) Double.POSITIVE_INFINITY else μ + Math.sqrt(α * Math.log(t.toDouble()) / (2 * depth * (t - 1)))
//    }
//
//    fun getLowerBoundsValue(μ: Double, t: Int, depth: Int, α: Double = 2.0): Double {
//        return if (t == 1) Double.POSITIVE_INFINITY else μ - Math.sqrt(α * Math.log(t.toDouble()) / (2 * depth * (t - 1)))
//    }
//
//    fun updateBounds(state : BeliefState){
//        mdp.addStates(mdp.generateStates(1, state))
//        val qValue = getMaxQ(state)
//        upperBound[state] = getUpperBoundsValue(qValue, state.leftSum(), state.totalSum())
//        lowerBound[state] = getLowerBoundsValue(qValue, state.leftSum(), state.totalSum())
//    }
//
//    fun runSampleTrial(initState: BeliefState, level: Int) : Double{
//        var state = initState
//        val stack = Stack<BeliefState>()
//        for (i in level..horizon-1){
//            stack.push(state)
//            updateBounds(state)
//
//            val (successorStates, successorValues) = getSuccessors(state) //b(y) in paper algorithm
//            val sumSuccessorValues = successorValues.sum()  //B in paper algorithm
//
//            if(sumSuccessorValues < ((upperBound[initState]!! - lowerBound[initState]!!)/T) ) break
//
//            for (i in 0..3) successorValues[i] = successorValues[i] / sumSuccessorValues
//
//            state = successorStates[ sampleSuccessor(successorValues) ]
//        }
//
//        while(!stack.isEmpty()){
//            state = stack.pop()
//            updateBounds(state)
//        }
//
//        return 0.0
//    }
//
//    fun simulate(currentState: BeliefState, level: Int) {
//        var count = 0
//        while ( runSampleTrial(currentState, level) > eps){
//            count++
//        }
//    }
//}
//
//fun brtdp(horizon: Int, world: Simulator, simulator: Simulator, rollOutCount: Int): List<Double>  {
//    val mdp = MDP(horizon + 1)
//    var currentState: BeliefState = mdp.startState
//    val averageRewards: MutableList<Double> = ArrayList(horizon)
//    val simulationCount = 200
//    val eps = 0.01  //Need to make sure about this value
//    val T = 50.0  //May need to tune this value by trial & error
//    var sum = 0.0
//
//    val brtdp = Brtdp(mdp, simulator, simulationCount, horizon, eps, T)
//
//    (0..horizon - 1).forEach { level ->
//        brtdp.simulate(currentState, level)
//        bellmanUtilityUpdate(currentState, mdp)
//        val (bestAction, bestReward) = selectBestAction(currentState, mdp)
//        val (nextState, reward) = world.transition(currentState, bestAction)
//
//        currentState = nextState
//        sum += reward
//        averageRewards.add(sum / (level + 1.0))
//    }
//
//    return averageRewards //Need to make sure about the return value & need to implement the online assumption
//}
//
//fun executeBrtdp(horizon: Int, world: Simulator, simulator: Simulator, probabilities: DoubleArray, iterations: Int): List<Result> {
//    val results: MutableList<Result> = ArrayList(iterations)
//    val rollOutCounts = intArrayOf(10, 100, 500, 1000)
//    val expectedMaxReward = probabilities.max()!!
//
//    brtdp(horizon, world, simulator, 100)
//
//    rollOutCounts.forEach { rollOutCount ->
//        val rewardsList = IntStream.range(0, iterations).mapToObj {
//            brtdp(horizon, world, simulator, rollOutCount)
//        }
//
//        val sumOfRewards = DoubleArray(horizon)
//        rewardsList.forEach { rewards ->
//            (0..horizon - 1).forEach {
//                sumOfRewards[it] = rewards[it] + sumOfRewards[it]
//            }
//        }
//
//        val averageRewards = sumOfRewards.map { expectedMaxReward - it / iterations }
//
//        results.add(Result("BRTDP$rollOutCount", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))
//    }
//
//    return results
//}
