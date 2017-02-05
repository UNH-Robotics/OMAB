package edu.unh.cs.ai.omab.algorithms

import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * @author Bence Cserna (bence@cserna.net)
 */
internal class OptimisticLookaheadKtTest {

    @Test
    fun testConstraintFilterLambda() {
        val good1 = listOf(1.0, 1.0, 1.0)
        val good2 = listOf(0.0, 0.0, 0.0)
        val good3 = listOf(0.2, 0.2, 0.2)
        val good4 = listOf(1.0, 0.9, 0.0)
        val good5 = listOf(0.3, 0.3, 0.2)
        val good6 = listOf(0.4, 0.3, 0.2)

        val bad1 = listOf(0.2, 0.3, 0.2)
        val bad2 = listOf(0.2, 0.3, 0.4)
        val bad3 = listOf(0.0, 0.5, 0.6)

        val goods = setOf(good1, good2, good3, good4, good5, good6)
        val bads = setOf(bad1, bad2, bad3)

        val all = goods + bads

        val filtered = all.filter { list -> list.indices.drop(1).all { list[it - 1] >= list[it] } }

        println("all")
        all.forEach(::println)
        println("filtered")
        filtered.forEach(::println)

        assertTrue(filtered.size == goods.size)
        assertTrue(filtered.all { it in goods })
    }


}