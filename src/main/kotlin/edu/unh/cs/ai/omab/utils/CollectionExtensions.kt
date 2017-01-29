package edu.unh.cs.ai.omab.utils

import java.util.*

/**
 * @author Bence Cserna (bence@cserna.net)
 */

inline fun <R, T: R> Iterable<T>.accumulate(operation: (R, T) -> R): List<R> {
    val destination = ArrayList<R>(10)

    val iterator = this.iterator()
    if (!iterator.hasNext()) throw UnsupportedOperationException("Empty collection can't be reduced.")
    var accumulator: R = iterator.next()
    while (iterator.hasNext()) {
        accumulator = operation(accumulator, iterator.next())
        destination.add(accumulator)
    }

    return destination
}

/**
 * Returns the first element yielding the largest value of the given function or `null` if there are no elements.
 */
inline fun <T, R : Comparable<R>> Iterable<T>.maxValueBy(selector: (T) -> R): R? {
    val iterator = iterator()
    if (!iterator.hasNext()) return null
    var maxElem = iterator.next()
    var maxValue = selector(maxElem)
    while (iterator.hasNext()) {
        val e = iterator.next()
        val v = selector(e)
        if (maxValue < v) {
            maxElem = e
            maxValue = v
        }
    }
    return maxValue
}