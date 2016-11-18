package edu.unh.cs.ai.omab.experiment

interface TerminationChecker {

    /**
     * @brief Called just before an agent starts planning / selecting an action 
     */
    fun init()

    /**
     * @brief Will terminate the agent action selection phase if it returns false
     */
    fun reachedTermination(): Boolean
}
