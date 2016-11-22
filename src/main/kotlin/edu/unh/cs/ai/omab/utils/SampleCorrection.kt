package edu.unh.cs.ai.omab.utils

import edu.unh.cs.ai.omab.domain.BeliefState
import org.apache.commons.math3.distribution.BetaDistribution
import java.util.*


fun sampleCorrection(state: BeliefState): DoubleArray {
    val alphas = state.alphas.copyOf()
    val betas = state.betas.copyOf()

    val newTransitionProbabilities = sampleFrom(alphas, betas)
    return newTransitionProbabilities
}

private fun sampleFrom(alphas: IntArray, betas: IntArray): DoubleArray {
    val sampleNumber = 10
    val random = Random()
    val distributions = (0..alphas.size - 1).map {
        BetaDistribution(alphas[it].toDouble(), betas[it].toDouble())
    }

    val samples = ArrayList<DoubleArray>()

    (0..sampleNumber).forEach {
        samples.add((0..distributions.size - 1).map {
            distributions[it].inverseCumulativeProbability(random.nextDouble())
        }.toDoubleArray())
    }

    val removeIndices = ArrayList<DoubleArray>()

    fun rankOrder(input: DoubleArray): Boolean {
        var predicate = true
        (0..input.size - 1).forEach { index ->
            if (index + 1 != input.size) {
                if (input[index] < input[index + 1]) {
                    predicate = false
                }
            }
        }
        return predicate
    }

    val filteredSamples = samples.filter(::rankOrder)


//    print("size before remove:"); println(samples.size)
    removeIndices.forEach { samples.remove(it) }
//    print("size after remove:"); println(samples.size)

    val newTransitions = ArrayList<Double>()


//    filteredSamples.forEach {
//        it.forEach(::print)
//        println()
//    }

    (0..alphas.size - 1).forEach { i ->
        var currentSum = 0.0
        (0..filteredSamples.size - 1).forEach { j ->
            currentSum += filteredSamples[j][i]
        }
        newTransitions.add(currentSum / (filteredSamples.size))
    }
//    println(newTransitions.size)
//    newTransitions.forEach { print("$it "); println() }

    return newTransitions.toDoubleArray()
}
