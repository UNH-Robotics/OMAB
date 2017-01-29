package edu.unh.cs.ai.omab.domain

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * @author Bence Cserna (bence@cserna.net)
 */
class BeliefStateTest {

    @Test
    fun isConsistentSingleArm() {
        assertTrue(BeliefState(intArrayOf(1), intArrayOf(1)).isConsistent())
        assertTrue(BeliefState(intArrayOf(2), intArrayOf(1)).isConsistent())
        assertTrue(BeliefState(intArrayOf(1), intArrayOf(2)).isConsistent())
    }

    @Test
    fun isConsistentTwoArms() {
        // Initial state: consistent
        assertTrue(BeliefState(intArrayOf(1, 1), intArrayOf(1, 1)).isConsistent())

        // Left arm looks better: consistent
        assertTrue(BeliefState(intArrayOf(2, 1), intArrayOf(1, 1)).isConsistent())
        assertTrue(BeliefState(intArrayOf(1, 1), intArrayOf(1, 2)).isConsistent())

        // Right arm looks better: inconsistent
        assertFalse(BeliefState(intArrayOf(2, 3), intArrayOf(1, 1)).isConsistent())
        assertFalse(BeliefState(intArrayOf(1, 2), intArrayOf(1, 1)).isConsistent())
        assertFalse(BeliefState(intArrayOf(1, 1), intArrayOf(2, 1)).isConsistent())
    }
}