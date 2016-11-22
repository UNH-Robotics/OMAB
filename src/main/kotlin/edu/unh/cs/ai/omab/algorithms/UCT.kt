package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.utils.maxIndex
import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator
import edu.unh.cs.ai.omab.experiment.Configuration
import edu.unh.cs.ai.omab.experiment.Result
import edu.unh.cs.ai.omab.experiment.TerminationChecker
import edu.unh.cs.ai.omab.experiment.terminationCheckers.FakeTerminationChecker
import java.util.*
import java.util.stream.IntStream

/**
 * Can apply UCT on any belief state given the current state and timestep
 *
 * Assumes the Bandit problem with 2 arms and is meant to solve the problem
 * where the expected reward of each arm is unknown and represented as a belief state
 */
class UCTPlanner(val numberOfActions: Int, val simulator: Simulator, val alpha: Double, val numSimulations: Int, val horizon: Int, val maxDepth: Int, val debug: Boolean = false) {

    private val random = Random()
    private var currentTimeStep = 0

    /**
     * A node in the UCT data structure
     *
     * qValues[i] returns the q value of action i
     * n[i] the number of times it has been taken
     */
    data class UCTNode(var qValues: DoubleArray, var n: IntArray) {

        /**
         * Initiates with zeros for q values and frequency of actions taken except for action i, which is set to n = 1 and q value of q
         */
        constructor(i: Int, q: Double, size: Int) : this(DoubleArray(size), IntArray(size)) {
            update(i, q)
        }

        /**
         * Increments N and updates the q value for the corresponding action
         */
        public fun update(action: Int, q: Double) {
            n[action]++
            qValues[action] += (q - qValues[action]) / n[action]
        }
    }

    /**
     * UCT stores a q value for each possible action at each <state,depth> pair.
     * Since the counts in the (belief) can only be equal at the same depth,
     * the beliefstate is always unique for its depth. So we can simply represent
     * the q values for the <depth, state> pair by storing it by state.
     */
    private var graph: MutableMap<BeliefState, UCTNode> = HashMap()

    /**
     * Returns true if either max depth or horizon has been reached
     */
    private fun exceedsMaxDepth(depth: Int) = depth + currentTimeStep >= horizon || depth >= maxDepth

    /**
     * The UCT rollout is a simulation where the actions are taken randomly
     * until the horizon has been reached
     * Adds the starting point node to the UCT datastructure
     */
    private fun rollout(state: BeliefState, depth: Int): Double {
        if (debug) print("In rollout starting at depth ${depth}\n")

        // base case
        if (exceedsMaxDepth(depth)) {
            return 0.0
        }

        val firstAction = random.nextInt(numberOfActions)
        var (nextState, rolloutReturn) = simulator.transition(state, firstAction)

        var currentDepth = depth + 1
        // rollout until horizon reached
        while (!exceedsMaxDepth(currentDepth)) { // TODO: add check whether state is terminal to generalize to other problems
            val action = random.nextInt(numberOfActions)
            val transitionResult = simulator.transition(nextState, action)

            nextState = transitionResult.state
            rolloutReturn += transitionResult.reward

            currentDepth++
        }

        // add new node to datastructure
        graph.put(state, UCTNode(firstAction, rolloutReturn.toDouble(), numberOfActions))

        if (debug) print("Ended rollout at depth ${currentDepth} with return action ${firstAction} for return of ${rolloutReturn.toDouble()} \n")
        return rolloutReturn.toDouble()
    }

    /**
     * Recursively traverses through the tree, selecting actions according to UCB and transition according to the simulator
     * Most importantly will stop when the horizon has been reached and changes into random rollouts when a leave has been reached
     *
     */
    private fun recurTreeSearch(state: BeliefState, depth: Int): Double {
        if (debug) print("In tree at depth ${depth}\n")

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

        uctNode.update(action, q)
        return q
    }

    /**
     * Returns the best action through UCB according to the uct node q and n values
     */
    private fun selectActionUcb(uctNode: UCTPlanner.UCTNode): Int {

        // total amount of times this uct node has been visited
        val N = uctNode.n.sum()

        // loop through all UCB values and keep highest
        var bestAction = 0
        var bestQ = upperConfidenceBoundsValue(uctNode.qValues[0], uctNode.n[0], N, (alpha * (horizon - currentTimeStep)))

        var i = 1
        while (i < numberOfActions) {

            var q = upperConfidenceBoundsValue(uctNode.qValues[i], uctNode.n[i], N, (alpha * (horizon - currentTimeStep)))

            if (q > bestQ) {
                bestAction = i;
                bestQ = q;
            }

            i++
        }

        if (debug) print("Selectin action ${bestAction} from node ${uctNode}\n")

        return bestAction;
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
            if (debug) print("Running simulation ${count} with root node ${graph[rootState]}\n")
            recurTreeSearch(rootState, 0)
        }
    }

    /**
     * Performs the actual tree search and returns the best action
     * See Kocsis, L., and Szepesvari, C. 2006 (Bandit Based Monte-Carlo Planning) for pseudo code
     */
    fun selectAction(rootState: BeliefState, timestep: Int, terminationChecker: TerminationChecker): Int {
        currentTimeStep = timestep

        buildTree(rootState, terminationChecker)

        val rootQNode = graph[rootState] ?:
                throw RuntimeException("UCT did not create any Q values associated with the rootState")

        if (debug) print("rootQNode before selecting action is ${rootQNode}\n")


        return rootQNode.qValues.maxIndex() ?: throw RuntimeException("Somehow rootNode was created but no q values were attached")
    }
}

/**
 * Applies UCT on the provided MDP
 */
fun executeUct(world: Simulator, simulator: Simulator, configuration: Configuration, debug: Boolean): List<Double> {
    val averageRewards: MutableList<Double> = ArrayList(configuration.horizon)
    var sum = 0.0

    /* UCT parameters */
    val terminationChecker = FakeTerminationChecker()
    val maxNumSimulations = 100
    val maxDepth = 20
    val alpha = 2500.0

    val planner = UCTPlanner(configuration.arms, simulator, alpha, maxNumSimulations, configuration.horizon, maxDepth, debug)
    var currentState = MDP(numberOfActions = configuration.arms).startState

    (0..configuration.horizon - 1).forEach { level ->
        // select action
        terminationChecker.init()
        val action = planner.selectAction(currentState, level, terminationChecker)

        // apply action
        if (debug) print("Taking action ${action} in real world on ${currentState}\n")
        val (nextState, reward) = world.transition(currentState, action)
        currentState = nextState
        sum += reward
        averageRewards.add(sum / (level + 1.0))
    }

    return averageRewards
}

/**
 * Executes UCT iteration times and returns an evaluation
 */
fun evaluateUct(world: Simulator, simulator: Simulator, probabilities: DoubleArray, configuration: Configuration): List<Result> {
    val debug = false

    val results: MutableList<Result> = ArrayList(configuration.iterations)
    val expectedMaxReward = probabilities.max()!!

    val rewardsList = IntStream.range(0, configuration.iterations).mapToObj {
        if (debug) print("Executing uct iteration ${it}\n")
        executeUct(world, simulator, configuration, debug)
    }

    val sumOfRewards = DoubleArray(configuration.horizon)
    rewardsList.forEach { rewards ->
        (0..configuration.horizon - 1).forEach {
            sumOfRewards[it] = rewards[it] + sumOfRewards[it]
        }
    }

    val averageRewards = sumOfRewards.map { expectedMaxReward - it / configuration.iterations }

    results.add(Result("UCT", probabilities, expectedMaxReward, averageRewards.last(), expectedMaxReward - averageRewards.last(), averageRewards))

    return results
}
