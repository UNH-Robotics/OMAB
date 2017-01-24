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