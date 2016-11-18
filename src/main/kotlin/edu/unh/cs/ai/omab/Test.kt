package edu.unh.cs.ai.omab

import edu.unh.cs.ai.omab.algorithms.calculateLookAhead
import edu.unh.cs.ai.omab.domain.BanditSimulator
import edu.unh.cs.ai.omab.domain.BanditWorld
import edu.unh.cs.ai.omab.domain.MDP
import edu.unh.cs.ai.omab.domain.Simulator

fun main(args: Array<String>) {
    println("Unit tests")
    var horizon: Int = 100
    val mdp: MDP = MDP(horizon)
    val banditSimulator: BanditSimulator = BanditSimulator()


    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 1.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 2.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 3.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 4.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 6.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 7.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 8.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 9.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 10.0)
    unitTest(mdp, horizon, BanditWorld(0.5, 0.6), banditSimulator, 11.0)
}

private fun unitTest(mdp: MDP, horizon: Int, world: Simulator, simulator: Simulator, numberOfStates: Double) {
    val lookAhead = calculateLookAhead(mdp, horizon, world, simulator, numberOfStates)
    print("$lookAhead ,")
}
