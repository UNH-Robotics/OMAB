package edu.unh.cs.ai.omab.experiment

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Configuration(
        val arms: Int,
        val rewards: DoubleArray,
        val horizon: Int,
        val experimentProbabilities: List<DoubleArray>,
        val iterations: Int)