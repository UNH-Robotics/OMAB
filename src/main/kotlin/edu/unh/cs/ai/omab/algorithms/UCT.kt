package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.Action.LEFT
import edu.unh.cs.ai.omab.domain.Action.RIGHT
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.Simulator
import org.apache.commons.math3.distribution.BetaDistribution
import java.util.stream.IntStream
import java.util.*

/**
* @brief 
* 
* Can apply UCT on any belief state given the current state and timestep
*
* Assumes the Bandit problem with 2 arms and is meant to solve the problem
* where the expected reward of each arm is unknown and represented as a belief state
* 
* @param simulator is the generative simulator that is used to step through the tree
* @param num_simulations is the amount of simulations to use while building the tree
* @param horizon is the horizon of the problem
*/
class UCTPlanner(val simulator: Simulator, val num_simulations: Int, val horizon: Int) {

    /**
    * @brief The current time step UCT is planning for
    */
    private var curTime = 0;

    /**
    * @brief A node in the UCT tree / graph
    *
    * Contains the q value of - and amount of times the action has been taken
    */
    data class UCTNode(var leftQ: Double, var leftN: Int, var rightQ: Double, var rightN: Int);

    /**
    * UCT stores a q value for each possible action at each <state,depth> pair. 
    * Since the counts in the (belief) can only be equal at the same depth,
    * the beliefstate is always unique for its depth. So we can simply represent 
    * the q values for the <depth, state> pair by storing it by state.
    */
    private var graph: MutableMap<BeliefState, UCTNode> = HashMap()
    
    private fun updateUCTNode(node: UCTNode, action: Action, q: Double) {
        // @TODO: update q values of node (incremental average)
        if (action == LEFT) {
            node.leftN++
        } else if (action == RIGHT) {
            node.rightN++
        } else {
            throw RuntimeException("UCT did not create any Q values associated with the rootState") 
        }
    }

    /**
    * @brief Recursively traverses through the tree, selecting actions according to UCB and transition according to the simulator
    *
    * Most importantly will stop when the horizon has been reached
    * Changes into random rollouts when a leave has been reached
    * Adds a node to the tree at the start of each rollout 
    * 
    * @param state the current state 
    * @param depth the current depth in the tree
    *
    * @return 
    */
    private fun recurTreeSearch(state: BeliefState, depth: Int): Double {

        // base case: reached end up horizon and return
        if (depth + curTime >= horizon) {
            return 0.0;
        }

        var uctNode = graph[state]

        // perform random rollouts if reached outside of UCT explored tree
        if (uctNode == null) {
            // @TODO: implement rollout
            return rollout(state, depth+1)
        }

        // still inside the tree: keep on recurring deeper
        // @TODO: implement selectActionUCB
        val action = selectActionUCB(uctNode);
        val (nextState, reward) = simulator.transition(state, action)
        val q = reward + recurTreeSearch(nextState, depth+1)

        updateUCTNode(uctNode, action, q)
        return q
    }

    /**
    * @brief Builds the UCT tree from root state assuming timestep 
    *
    * @param rootState the root state of the tree
    *
    * @return void
    */
    private fun buildTree(rootState: BeliefState) {
        assert(curTime < horizon)

        // start empty
        graph.clear()

        var count = 0;
        while (count++ < num_simulations) {
            recurTreeSearch(rootState, 0)
        }
    }

    /**
    * @brief Performs the actual tree search and returns the best action
    *
    * @param rootState the starting state
    * @param simulator the simulator to use for steps
    * @param timestep is the current timestep
    *
    * See iKocsis, L., and Szepesvari, C. 2006 (Bandit Based Monte-Carlo Planning) for pseudo code 
    * @return an action
    */
    public fun selectAction(rootState: BeliefState, timestep: Int): Action {
        curTime = timestep

        buildTree(rootState);

        // @TODO: does this have to be so ugly?
        val rootQNode = graph[rootState] ?: 
                        throw RuntimeException("UCT did not create any Q values associated with the rootState") 

        return if (rootQNode.leftQ > rootQNode.rightQ) LEFT else RIGHT;
    }

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

    // @TODO: get some actual way of determine when to terminate UCT
    var num_simulations = 1000

    var currentState = mdp.startState
    var planner = UCTPlanner(simulator, num_simulations, horizon);

    return IntStream.iterate(0, {t -> t + 1}).limit(horizon.toLong()).mapToLong {
        // select action
        val action = planner.selectAction(currentState, it);

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
