package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.calculateLookAhead
import edu.unh.cs.ai.omab.domain.*
import java.util.*

fun main(args: Array<String>) {
    println("Unit tests")
    var horizon: Int = 100
    val mdp: MDP = MDP(horizon, 3)
    val banditSimulator: BanditSimulator = BanditSimulator(rewards = doubleArrayOf(1.0, 1.0, 1.0))

    unitTest(mdp, horizon, BanditWorld(doubleArrayOf(0.8,0.3,0.5)), banditSimulator, 1.0)

    val levelList = mdp.generateStatess(0, 3, BeliefState(intArrayOf(1,1,1), intArrayOf(1,1,1)))
    println(levelList.size)
    println(levelList)

}

private fun unitTest(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator, numberOfStates: Double) {
    val lookAhead = calculateLookAhead(horizon)
//    print("$lookAhead ,")
}
