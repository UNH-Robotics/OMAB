package edu.unh.cs.ai.omab.utils

/**
 * @author Bence Cserna (bence@cserna.net)
 */
infix fun Double.pow(b: Double) = Math.pow(this, b)
infix fun Int.pow(b: Double) = Math.pow(this.toDouble(), b)
infix fun Double.pow(b: Int) = Math.pow(this, b.toDouble())
