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
     * @brief random double generator
     */
    val random = Random()

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
        when (action) {
            LEFT -> {
                node.leftN++
                node.leftQ += (q - node.leftQ) / node.leftN
            }
            RIGHT -> {
                node.rightN++
                node.rightQ += (q - node.rightQ) / node.rightN
            }
            else -> throw RuntimeException("UCT did not create any Q values associated with the rootState")
        }
    }

    /**
     * @brief checks if horizon is reached
     *
     * @TODO make sure it is not > instead of >=
     */
    private fun reachedHorizon(depth: Int) = depth + currentTimeStep >= horizon

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
        println("Starting rollout at depth $depth")
        // base case: reached horizon
        if (reachedHorizon(depth)) {
            println("Reached depth")
            return 0.0
        }

        val firstAction = if (random.nextBoolean()) LEFT else RIGHT
        var (nextState, rolloutReturn) = simulator.transition(state, firstAction)

        var currentDepth = depth + 1
        // TODO: add check whether state is terminal to generalize to other problems
        // rollout until horizon reached
        while (!reachedHorizon(currentDepth)) {
            println("In rollout at depth $currentDepth")
            val action = if (random.nextBoolean()) LEFT else RIGHT
            val transitionResult = simulator.transition(nextState, action)
            nextState = transitionResult.state

            rolloutReturn += transitionResult.reward
            currentDepth++
        }

        // create and add new node to datastructure
        val newNode = when (firstAction) {
            LEFT -> UCTNode(rolloutReturn.toDouble(), 1, 0.0, 0)
            RIGHT -> UCTNode(0.0, 0, rolloutReturn.toDouble(), 1)
            else -> throw RuntimeException("UCT did not create any Q values associated with the rootState")
        }

        graph.put(state, newNode)

        println("Ended rollout at depth $currentDepth")
        return rolloutReturn.toDouble()
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
        println("In tree at depth $depth")

        // base case: return if horizon has been reached
        if (reachedHorizon(depth)) {
            return 0.0
        }

        val uctNode: UCTNode = graph[state] ?: return rollout(state, depth + 1)

        // perform random rollouts if reached outside of UCT explored tree

        // still inside the tree: keep on recurring deeper
        val action = selectActionUcb(uctNode)
        val (nextState, reward) = simulator.transition(state, action)
        val q = reward + recurTreeSearch(nextState, depth + 1)

        updateUCTNode(uctNode, action, q)
        return q
    }

    private fun selectActionUcb(uctNode: UCTPlanner.UCTNode): Action {
        val depth = uctNode.leftN + uctNode.rightN
        val leftUcbValue = upperConfidenceBoundsValue(uctNode.leftQ, uctNode.leftN, depth)
        val rightUcbValue = upperConfidenceBoundsValue(uctNode.rightQ, uctNode.rightN, depth)

        return if (leftUcbValue > rightUcbValue) LEFT else RIGHT
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
            println("Running simulation $count")
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
     * See Kocsis, L., and Szepesvari, C. 2006 (Bandit Based Monte-Carlo Planning) for pseudo code
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
 * @param mdp
 * @param Int
 * @param Simulator
 * @param
 *
 * @return accumulated reward of a single instance of UCT planning on this problem
 */
fun uct(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Long {

    // @TODO: get some actual way of determine when to terminate UCT
    val numSimulations = 1000

    var currentState = mdp.startState
    val planner = UCTPlanner(simulator, numSimulations, horizon)

    return IntStream.iterate(0, { t -> t + 1 }).limit(horizon.toLong()).mapToLong {
        println("Running timestep $it")
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
