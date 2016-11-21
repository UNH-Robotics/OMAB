package edu.unh.cs.ai.omab.experiment.terminationCheckers

import edu.unh.cs.ai.omab.experiment.TerminationChecker

/**
 * Will never fail the test, for debugging purposes
 */
class FakeTerminationChecker : TerminationChecker {

    override fun init() {
    }

    /**
     * @brief Will never terminate.
     */
    override fun reachedTermination() = false
}
