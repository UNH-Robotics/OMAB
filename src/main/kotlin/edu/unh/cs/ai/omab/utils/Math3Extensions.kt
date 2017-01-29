package edu.unh.cs.ai.omab.utils

import edu.unh.cs.ai.omab.domain.BeliefState
import org.apache.commons.math3.distribution.BetaDistribution

/**
 * @author Bence Cserna (bence@cserna.net)
 */
fun BeliefState.betaDistributions() = alphas.indices.map { BetaDistribution(this.alphas[it].toDouble(), this.betas[it].toDouble()) }
