package edu.unh.cs.ai.omab.algorithms

import edu.unh.cs.ai.omab.domain.BeliefState
import edu.unh.cs.ai.omab.experiment.Configuration
import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */

fun parseGittinsIndices(path: String): HashMap<Pair<Int, Int>, Double> {
    val gittinsIndices = hashMapOf<Pair<Int, Int>, Double>()
    val input = Unit::class.java.classLoader.getResourceAsStream(path) ?: throw RuntimeException("Resource not found")
    input.reader().readLines()
            .drop(1) // The first line contains the headers
            .map { it.split(",") }
            .filter { it.size == 3 }
            .forEach { gittinsIndices[it[0].toInt() to it[1].toInt()] = it[2].toDouble() }

    return gittinsIndices
}

val GITTINS_INDICES = parseGittinsIndices("gittins.csv")

fun gittinsIndex(state: BeliefState, configuration: Configuration, random: Random): Int {
    return state.arms.maxBy { GITTINS_INDICES[state.alphas[it] to state.betas[it]]!! * configuration.rewards[it] }!!
}