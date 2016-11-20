package edu.unh.cs.ai.omab.algorithms

import java.util.*
import java.util.stream.IntStream

import edu.unh.cs.ai.omab.experiment.TerminationChecker
import edu.unh.cs.ai.omab.experiment.terminationCheckers.FakeTerminationChecker
import edu.unh.cs.ai.omab.experiment.terminationCheckers.TimeTerminationChecker

import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.domain.BeliefState

import edu.unh.cs.ai.omab.domain.Action
import edu.unh.cs.ai.omab.domain.Action.LEFT
import edu.unh.cs.ai.omab.domain.Action.RIGHT

/**
 * Can apply UCT on any belief state given the current state and timestep
 *
 * Assumes the Bandit problem with 2 arms and is meant to solve the problem
 * where the expected reward of each arm is unknown and represented as a belief state
 */
class UCTPlanner(val simulator: Simulator, val numSimulations: Int, val horizon: Int, val maxDepth: Int) {

    private val random = Random()
    private var currentTimeStep = 0

    /**
     * TODO: generalize to N <q,n> pairs for N is the amount of possible actions.
     */
    data class UCTNode(var leftQ: Double, var leftN: Int, var rightQ: Double, var rightN: Int)

    /**
     * UCT stores a q value for each possible action at each <state,depth> pair.
     * Since the counts in the (belief) can only be equal at the same depth,
     * the beliefstate is always unique for its depth. So we can simply represent
     * the q values for the <depth, state> pair by storing it by state.
     */
    private var graph: MutableMap<BeliefState, UCTNode> = HashMap()

    private fun updateUCTNode(node: UCTNode, action: Action, q: Double) {
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
    * @brief Returns true if either max depth or horizon has been reached
    */
    private fun exceedsMaxDepth(depth: Int) = depth + currentTimeStep >= horizon || depth >= maxDepth

    /**
     * The UCT rollout is a simulation where the actions are taken randomly until the horizon has been reached
     * Adds the starting point node to the UCT datastructure
     */
    private fun rollout(state: BeliefState, depth: Int): Double {
        // base case
        if (exceedsMaxDepth(depth)) {
            return 0.0
        }

        val firstAction = if (random.nextBoolean()) LEFT else RIGHT
        var (nextState, rolloutReturn) = simulator.transition(state, firstAction)

        var currentDepth = depth + 1
        // TODO: add check whether state is terminal to generalize to other problems
        // rollout until horizon reached
        while (!exceedsMaxDepth(currentDepth)) {
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

        print("Ended rollout at depth ${currentDepth}\n")
        return rolloutReturn.toDouble()
    }

    /**
     * Recursively traverses through the tree, selecting actions according to UCB and transition according to the simulator
     * Most importantly will stop when the horizon has been reached and changes into random rollouts when a leave has been reached
     *
     */
    private fun recurTreeSearch(state: BeliefState, depth: Int): Double {

        // base case
        if (exceedsMaxDepth(depth)) {
            return 0.0
        }

        // perform random rollouts if reached outside of UCT explored tree
        val uctNode: UCTNode = graph[state] ?: return rollout(state, depth)

        // still inside the tree: keep on recurring deeper
        val action = selectActionUcb(uctNode)
        val (nextState, reward) = simulator.transition(state, action)
        val q = reward + recurTreeSearch(nextState, depth + 1)

        updateUCTNode(uctNode, action, q)
        return q
    }

    private fun selectActionUcb(uctNode: UCTPlanner.UCTNode): Action {
        val N = uctNode.leftN + uctNode.rightN
        val leftUcbValue = upperConfidenceBoundsValue(uctNode.leftQ, uctNode.leftN, N, (10000.0 * (horizon - currentTimeStep)))
        val rightUcbValue = upperConfidenceBoundsValue(uctNode.rightQ, uctNode.rightN, N, (10000.0 * (horizon - currentTimeStep)))

        return if (leftUcbValue > rightUcbValue) LEFT else RIGHT
    }

    /**
     * Builds the UCT datastructure from root state assuming timestep
     */
    private fun buildTree(rootState: BeliefState, terminationChecker: TerminationChecker) {
        assert(!exceedsMaxDepth(0))

        // start empty
        graph.clear()

        var count = 0
        while (count++ < numSimulations && !terminationChecker.reachedTermination()) {
            /*print("At simulation ${count}\n")*/
            recurTreeSearch(rootState, 0)
        }
    }

    /**
     * Performs the actual tree search and returns the best action
     * See Kocsis, L., and Szepesvari, C. 2006 (Bandit Based Monte-Carlo Planning) for pseudo code
     */
    fun selectAction(rootState: BeliefState, timestep: Int, terminationChecker: TerminationChecker): Action {
        currentTimeStep = timestep

        buildTree(rootState, terminationChecker)

        val rootQNode = graph[rootState] ?:
                throw RuntimeException("UCT did not create any Q values associated with the rootState")

        return if (rootQNode.leftQ > rootQNode.rightQ) LEFT else RIGHT
    }
}

/**
 * Applies UCT on the provided MDP
 */
fun uct(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator): Double {

    // has to be high enough to do at least one simulation..
    val terminationChecker = TimeTerminationChecker(100000)
    val maxNumSimulations = 1000000
    val maxDepth = 5

    var currentState = mdp.startState
    val planner = UCTPlanner(simulator, maxNumSimulations, horizon, maxDepth)

    return IntStream.iterate(0, { t -> t + 1 }).limit(horizon.toLong()).mapToLong {
        // select action
        terminationChecker.init()
        val action = planner.selectAction(currentState, it, terminationChecker)

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
    }.sum().toDouble()
}
