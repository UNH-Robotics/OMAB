package edu.unh.cs.ai.omab.experiment

import com.sun.org.apache.xpath.internal.operations.Bool

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class Configuration(
        val arms: Int,
        val rewards: DoubleArray,
        val horizon: Int,
        val experimentProbabilities: List<DoubleArray>,
        val iterations: Int,
        val specialSauce: Boolean)