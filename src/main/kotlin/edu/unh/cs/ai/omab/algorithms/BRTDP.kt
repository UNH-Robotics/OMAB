//package edu.unh.cs.ai.omab.algorithms
//
//import edu.unh.cs.ai.omab.domain.*
//import java.util.*
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
//        for(action in Action.getActions()){
//            for(isSuccess in listOf(true, false)){
//                val nextState = state.nextState(action, isSuccess)
//                successorStates.add(nextState)
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
//        for(i in 0..4){
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
//        val qValue = getMaxQ(state)
//        upperBound[state] = getUpperBoundsValue(qValue, state.leftSum(), state.totalSum())
//        lowerBound[state] = getLowerBoundsValue(qValue, state.leftSum(), state.totalSum())
//    }
//
//    fun runSampleTrial(initState: BeliefState) : Double{
//        var state = initState
//        val stack = Stack<BeliefState>()
//
//        while(true){
//            stack.push(state)
//            mdp.addStates(mdp.generateStates(1, state))
//            updateBounds(state)
//
//            val (successorStates, successorValues) = getSuccessors(state) //b(y) in paper algorithm
//            val sumSuccessorValues = successorValues.sum()  //B in paper algorithm
//
//            for (i in 0..4) successorValues[i] = successorValues[i] / sumSuccessorValues
//
//            if(sumSuccessorValues < ((upperBound[initState]!! - lowerBound[initState]!!)/T) ) break
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
//    fun simulate(currentState: BeliefState) {
//        var count = 0
//        while ( runSampleTrial(currentState) > eps){
//            count++
//        }
//        //(0..simulationCount).forEach { rollOut(currentState, currentDepth) }
//    }
//}
//
//fun brtdp(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
//    var currentState: BeliefState = mdp.startState
//    val localMDP = MDP(horizon + 1)
//    val simulationCount = 200
//    val eps = 0.01  //Need to make sure about this value
//    val T = 50.0
//
//    val brtdp = Brtdp(localMDP, simulator, simulationCount, horizon, eps, T)
//
//    brtdp.simulate(currentState)
//
//    return 0.0 //Need to make sure about the return value & need to implement the online assumption
//}