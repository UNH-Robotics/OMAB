package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.Action.LEFT
import edu.unh.cs.ai.omab.domain.Action.RIGHT
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import java.util.*
import java.util.stream.IntStream

/**
 * @brief
 *
 * Can apply UCT on any belief state given the current state and timestep
 *
 * Assumes the Bandit problem with 2 arms and is meant to solve the problem
 * where the expected reward of each arm is unknown and represented as a belief state
 *
 * @param simulator is the generative simulator that is used to perform the steps 
 * @param num_simulations is the amount of simulations used by UCT
 * @param horizon is the horizon of the problem
 */
class UCTPlanner(val simulator: Simulator, val num_simulations: Int, val horizon: Int) {

    /**
     * @brief The current time step UCT is planning for
     */
    private var currentTimeStep = 0

    /**
     * @brief A node in the UCT datastructure
     *
     * Contains the q value of - and amount of times the action has been taken
     *
     * @TODO: generalize to N <q,n> pairs for N is the amount of possible actions.
     */
    data class UCTNode(var leftQ: Double, var leftN: Int, var rightQ: Double, var rightN: Int)

    /**
     * UCT stores a q value for each possible action at each <state,depth> pair.
     * Since the counts in the (belief) can only be equal at the same depth,
     * the beliefstate is always unique for its depth. So we can simply represent
     * the q values for the <depth, state> pair by storing it by state.
     */
    private var graph: MutableMap<BeliefState, UCTNode> = HashMap()

    /**
     * @brief Updates a node of the UCT datastructure given a taken action and perceived q value
     *
     * Increments the count associated with the action
     * Calculates the new mean for the expected return of the associated action
     *
     * @TODO: make sure the provided node is a reference, not a copy
     *
     * @param node the node to update
     * @param action taken action
     * @param q the return from a single simulation from the node after taking action
     *
     * @return void
     */
    private fun updateUCTNode(node: UCTNode, action: Action, q: Double) {
        // @TODO: I'm sure kotlin has some nicer syntax than this...
        if (action == LEFT) {
            node.leftN++
            node.leftQ += (q - node.leftQ) / leftN
        } else if (action == RIGHT) {
            node.rightN++
            node.rightQ += (q - node.rightQ) / rightN
        } else {
            throw RuntimeException("UCT did not create any Q values associated with the rootState")
        }
    }

    /**
    * @brief The UCT rollout is a simulation where the actions are taken randomly until the horizon has been reached
    *
    * Adds the starting point node to the UCT datastructure 
    *
    * @param state the starting state
    * @param depth the starting depth from the search tree
    *
    * @return void
    */
    private fun rollout(state: BeliefState, depth: Int): Double {

        // base case: reached horizon
        if (reachedHorizon(depth)){
            return 0
        }

        val FirstAction = if (Random() < 0.50) LEFT else RIGHT
        val (nextState, rolloutReturn) = simulator.transition(state, action)

        // @TODO: add check whether state is terminal to generlize to other problems
        // rollout until horizon reached
        while (reachedHorizon(++depth)) {
            val action = if (Random() < 0.50) LEFT else RIGHT
            val (nextState, reward) = simulator.transition(state, action)
            rolloutReturn += reward;
        }

        // create and add new node to datastructure
        newNode = if (FirstAction == LEFT) {
            UCTNode(rolloutReturn, 1, 0, 0)
        } else if (FirstAction == RIGHT) {
            UCTNode(0, 0, rolloutReturn, 1)
        } else {
            throw RuntimeException("UCT did not create any Q values associated with the rootState")
        }
        graph.put(state, newNode)
    }

    /**
     * @brief Recursively traverses through the tree, selecting actions according to UCB and transition according to the simulator
     *
     * Most importantly will stop when the horizon has been reached
     * Changes into random rollouts when a leave has been reached
     * Adds a node to the datastructure at the start of each rollout
     *
     * @param state the current state
     * @param depth the current depth in the datastructure
     *
     * @return
     */
    private fun recurTreeSearch(state: BeliefState, depth: Int): Double {

        // base case: return if horizon has been reached
        if (reachedHorizon(depth)) {
            return 0.0
        }

        // @TODO: make sure this is a reference (not a copy)
        var uctNode = graph[state]

        // perform random rollouts if reached outside of UCT explored tree
        if (uctNode == null) {
            // @TODO: implement rollout
            return rollout(state, depth + 1)
        }

        // still inside the tree: keep on recurring deeper
        // @TODO: implement selectActionUCB
        val action = selectActionUCB(uctNode)
        val (nextState, reward) = simulator.transition(state, action)
        val q = reward + recurTreeSearch(nextState, depth + 1)

        updateUCTNode(uctNode, action, q)
        return q
    }

    /**
     * @brief Builds the UCT datastructure from root state assuming timestep
     *
     * @param rootState the root state of the datastructure
     *
     * @return void
     */
    private fun buildTree(rootState: BeliefState) {
        assert(!reachedHorizon(0))

        // start empty
        graph.clear()

        var count = 0
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
    fun selectAction(rootState: BeliefState, timestep: Int): Action {
        currentTimeStep = timestep

        buildTree(rootState)

        val rootQNode = graph[rootState] ?:
                throw RuntimeException("UCT did not create any Q values associated with the rootState")

        return if (rootQNode.leftQ > rootQNode.rightQ) LEFT else RIGHT
    }

}

/**
 * @brief Applies UCT on the provided MDP
 *
 * @TODO Currently will apply search all the way to the horizon, may make use of a max depth in future
 * @TODO documentation on parameters
 *
 * @param MDP
 * @param Int
 * @param Simulator
 *
 * @return accumulated reward of a single instance of UCT planning on this problem 
 */
fun uct(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Long {

    // @TODO: get some actual way of determine when to terminate UCT
    var num_simulations = 1000

    var currentState = mdp.startState
    var planner = UCTPlanner(simulator, num_simulations, horizon)

    return IntStream.iterate(0, { t -> t + 1 }).limit(horizon.toLong()).mapToLong {
        // select action
        val action = planner.selectAction(currentState, it)

        // apply action
        val (nextState, reward) = if (action == LEFT) {
            world.transition(currentState, LEFT)
        } else if (action == RIGHT) {
            world.transition(currentState, RIGHT)
        } else {
            throw RuntimeException("Impossible action selected")
        }

        currentState = nextState

        reward.toLong()
    }.sum()
}
