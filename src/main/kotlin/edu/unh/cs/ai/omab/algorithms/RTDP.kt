package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.io.File
import java.io.PrintWriter
import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Rtdp(val simulator: Simulator, val simulationCount: Int, val horizon: Int) {
    private var graph: MutableMap<BeliefState, UCTPlanner.UCTNode> = HashMap()

    data class Node(val qValue: Double)

    lateinit var maxAction : Action
    var maxValue = 0.0
    fun selectAction(currentState: BeliefState, it: Int, writer: PrintWriter): ActionResult {
        writer.append("current state: $currentState\n")
        currentTimeStep = it
        for(act in Action.getActions()){
            var actValue = 0.0
            for(isSuccess in listOf(true, false)) {
                val nextState = currentState.nextState(act, isSuccess)
                val rollVal = rollout(nextState, it, writer)

                writer.append("action: $act, next state: $nextState, rollVal: $rollVal\n")

                if(isSuccess) actValue += currentState.leftMean() * (nextState.utility+1) + rollVal
                else actValue += (1-currentState.leftMean()) * nextState.utility + rollVal
            }

            if(maxValue<actValue){
                maxValue = actValue
                maxAction = act
            }

            writer.append("actValue: $actValue, maxValue: $maxValue, maxAction: $maxAction\n")
            //println("actValue: $actValue, maxValue: $maxValue, maxAction: $maxAction")
        }
        return ActionResult(maxAction,maxValue)
        //throw UnsupportedOperationException("not implemented")
    }

    private var currentTimeStep = 0
    private val random = Random()
    private fun hasReachedHorizon(depth: Int) = depth >= horizon
    /**
     * The RTDP rollout is a simulation where the actions are taken based ont the beta distribution until the horizon has been reached
     */
    private fun rollout(state: BeliefState, depth: Int, writer: PrintWriter): Double {
        //println("depth: $depth, reachedHorizon: ${hasReachedHorizon(depth)}")
        // base case
        rollCount++;
        if (hasReachedHorizon(depth)) {
            return 0.0
        }
        //println("rollout called")
        val firstAction = if (random.nextBoolean()) Action.LEFT else Action.RIGHT
        var (nextState, rolloutReturn) = simulator.transition(state, firstAction)

        var currentDepth = depth + 1
        // TODO: add check whether state is terminal to generalize to other problems
        // rollout until horizon reached
        while (!hasReachedHorizon(currentDepth)) {
            val action = if (random.nextBoolean()) Action.LEFT else Action.RIGHT
            val transitionResult = simulator.transition(nextState, action)
            nextState = transitionResult.state

            rolloutReturn += transitionResult.reward
            //writer.append("Rollout, transition result: ${transitionResult.reward}")

            currentDepth++
        }
        return rolloutReturn.toDouble()
    }
}

data class ActionResult(val action: Action, val utility: Double)
private var rollCount = 0
fun rtdp(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {
    val simulationCount = 50

    var currentState = mdp.startState
    val rtdp = Rtdp(simulator, simulationCount, horizon)
    val writer = PrintWriter("RTDP_output.txt")

    val totalReward = (0..horizon).map { it->
        //println("horizon it: $it")
        val actionResult = rtdp.selectAction(currentState, it, writer)
        currentState.utility = actionResult.utility

        val (nextState, reward) = world.transition(currentState, actionResult.action)
        currentState = nextState

        reward.toDouble()

    }.sum()
    writer.append("rollCount: $rollCount")
    writer.close()

    return totalReward
}