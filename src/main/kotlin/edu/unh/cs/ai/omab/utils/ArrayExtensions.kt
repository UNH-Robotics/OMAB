package edu.unh.cs.ai.omab.utils

/**
 * Returns the index of largest element or `null` if there are no elements.
 */
fun IntArray.maxIndex(): Int? {
    if (isEmpty()) return null
    var max = this[0]
    var index = 0
    for (i in 1..lastIndex) {
        val e = this[i]
        if (max < e) {
            max = e
            index = i
        }
    }
    return index
}

/**
 * Returns the index of largest element or `null` if there are no elements.
 */
fun DoubleArray.maxIndex(): Int? {
    if (isEmpty()) return null
    var max = this[0]
    var index = 0
    for (i in 1..lastIndex) {
        val e = this[i]
        if (max < e) {
            max = e
            index = i
        }
    }
    return index
}