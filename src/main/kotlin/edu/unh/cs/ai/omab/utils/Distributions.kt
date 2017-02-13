package edu.unh.cs.ai.omab.utils

import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.random.RandomGenerator

class UncommonDistributions(private val random: RandomGenerator = MersenneTwister()) {

    companion object {
        val LOG4 = 1.3862943611198906
        val LOG4_5 = 1.5040773967762742
    }

    // =============== start of BSD licensed code. See LICENSE.txt
    /**
     * Returns a double sampled according to this distribution. Uniformly fast for all k > 0. (Reference:
     * Non-Uniform Random Variate Generation, Devroye http://cgm.cs.mcgill.ca/~luc/rnbookindex.html) Uses
     * Cheng's rejection algorithm (GB) for k>=1, rejection from Weibull distribution for 0 < k < 1.
     */
    fun rGamma(k: Double, lambda: Double): Double {
        var accept = false
        return if (k >= 1.0) {
            // Cheng's algorithm
            val b = k - LOG4
            val c = k + Math.sqrt(2.0 * k - 1.0)
            val lam = Math.sqrt(2.0 * k - 1.0)
            val cheng = 1.0 + LOG4_5
            var x: Double
            do {
                val u = random.nextDouble()
                val v = random.nextDouble()
                val y = 1.0 / lam * Math.log(v / (1.0 - v))
                x = k * Math.exp(y)
                val z = u * v * v
                val r = b + c * y - x
                if (r >= 4.5 * z - cheng || r >= Math.log(z)) {
                    accept = true
                }
            } while (!accept)
            x
        } else {
            // Weibull algorithm
            val c = 1.0 / k
            val d = (1.0 - k) * Math.pow(k, k / (1.0 - k))
            var x: Double
            do {
                val u = random.nextDouble()
                val v = random.nextDouble()
                val z = -Math.log(u)
                val e = -Math.log(v)
                x = Math.pow(z, c)
                if (z + e >= d + x) {
                    accept = true
                }
            } while (!accept)
            x
        } / lambda
    }

    // ============= end of BSD licensed code

    /**
     * Returns a random sample from a beta distribution with the given shapes
     * @param shape1 a double representing shape1
     * @param shape2 a double representing shape2
     *
     * @return a Vector of samples
     */
    fun rBeta(shape1: Double, shape2: Double): Double {
        val gam1 = rGamma(shape1, 1.0)
        val gam2 = rGamma(shape2, 1.0)
        return gam1 / (gam1 + gam2)
    }
}
