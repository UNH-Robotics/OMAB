package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Rtdp(val simulator: Simulator, val simulationCount: Int, val horizon: Int) {
    private var graph: MutableMap<BeliefState, UCTPlanner.UCTNode> = HashMap()

    data class Node(val qValue: Double)

    lateinit var maxAction : Action
    var maxValue = 0.0
    fun selectAction(currentState: BeliefState, it: Int): ActionResult {
        for(act in Action.getActions()){
            var actValue = 0.0
            var isSuccess = true
            for(i in 1..2) {
                val nextState = currentState.nextState(act, isSuccess)
                if(isSuccess) actValue += currentState.leftMean() * (nextState.utility+1)
                else actValue += (1-currentState.leftMean()) * nextState.utility
                isSuccess = false
            }

            if(maxValue<actValue){
                maxValue = actValue
                maxAction = act
            }
        }
        return ActionResult(maxAction,maxValue)
        throw UnsupportedOperationException("not implemented")
    }
}

data class ActionResult(val action: Action, val utility: Double)

fun rtdp(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Long {
    val simulationCount = 50

    var currentState = mdp.startState
    val rtdp = Rtdp(simulator, simulationCount, horizon)

    return (0..horizon).map {
        val actionResult = rtdp.selectAction(currentState, it)
        currentState.utility = actionResult.utility

        val (nextState, reward) = world.transition(currentState, actionResult.action)
        currentState = nextState

        reward.toLong()
    }.sum()
}