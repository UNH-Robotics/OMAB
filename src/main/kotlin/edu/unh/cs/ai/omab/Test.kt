package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.calculateLookAhead
import edu.unh.cs.ai.omab.domain.*
import kotlin.system.measureTimeMillis

fun main(args: Array<String>) {
    println("Unit tests")
    var horizon: Int = 100
    val mdp: MDP = MDP(horizon, 3)
    val banditSimulator: BanditSimulator = BanditSimulator(rewards = doubleArrayOf(1.0, 1.0, 1.0))

    unitTest(mdp, horizon, BanditWorld(doubleArrayOf(0.8, 0.2, 0.2)), banditSimulator, 1.0)
    var number = 4

    (0..number).forEach { stateNumbers ->
        val executionTime = measureTimeMillis {

            val levelList = mdp.generateStates(stateNumbers, BeliefState(intArrayOf(1, 1, 1), intArrayOf(1, 1, 1)))
            print("states generated ")
            println(levelList.size)

            println(levelList)
        }
        println("executionTime $executionTime")
        number += 1
    }


//    val nextLevel = mdp.generateNextLevel(BeliefState(intArrayOf(1,1,1), intArrayOf(1,1,1)))
//    println(nextLevel)

}

private fun unitTest(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator, numberOfStates: Double) {
    val lookAhead = calculateLookAhead(horizon)
//    print("$lookAhead ,")
}
