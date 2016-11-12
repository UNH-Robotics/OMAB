package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action.LEFT
import edu.unh.cs.ai.omab.domain.Action.RIGHT
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.Simulator
import org.apache.commons.math3.distribution.BetaDistribution
import java.util.stream.IntStream
import java.util.*

class UCTPlanner(val num_simulations: Int, val horizon: Int) {

    /**
    * @brief The q values associated with the actions in each node of the uct graph
    */
    data class QValues(var qLeft: Double, var qRight: Double);

    /**
    * @brief Performs the actual tree search and returns the best action
    *
    * @param rootState the starting state
    * @param simulator the simulator to use for steps
    * @param timestep is the current timestep
    * @param horizon is the maximum depth to plan to
    *
    * @return an action
    */
    public fun selectAction(rootState: BeliefState, simulator: Simulator, timestep: Int, horizon: Int) = LEFT

    private var graph: Map<BeliefState, QValues> = HashMap()
}

/**
* @brief Applies UCT on the provided MDP
*
* Currently will apply depth of the tree search up till horizon
* @param MDP
* @param Int
* @param Simulator
*
* @return return
*/
fun uct(mdp: MDP, horizon: Int, simulator: Simulator): Long {

    // @TODO: get some actual way of determine when to terminate uct
    var num_simulations = 1000

    var currentState = mdp.startState
    var planner = UCTPlanner(num_simulations, horizon);

    return IntStream.iterate(0, {t -> t + 1}).limit(horizon.toLong()).mapToLong {
        // select action
        val action = planner.selectAction(currentState, simulator, it, horizon);

        // apply action
        val (nextState, reward) = if (action == LEFT) {
            simulator.transition(currentState, LEFT)
        } else if (action == RIGHT){
            simulator.transition(currentState, RIGHT)
        } else {
            throw RuntimeException("Impossible action selected")
        }

        currentState = nextState

        reward.toLong()
    }.sum()
}
