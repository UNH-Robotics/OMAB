package edu.unh.cs.ai.omab.experiment.terminationCheckers

import edu.unh.cs.ai.omab.experiment.TerminationChecker
import java.util.concurrent.TimeUnit

 /**
  * @brief terminates if the provided time deadline has (almost) been reached
  * 
  * Notice this will set the starting time at creation and init()!
  */
class TimeTerminationChecker(val timeLimitMiliSeconds: Long) : TerminationChecker {

    private val timeLimit = TimeUnit.NANOSECONDS.convert(timeLimitMiliSeconds, TimeUnit.MICROSECONDS)
    private var startTime = System.nanoTime()

    /**
     * @brief to ensure the time limit is never 
     */
    private val epsilon = TimeUnit.NANOSECONDS.convert(2000, TimeUnit.MICROSECONDS)

    /**
     * @brief Sets the starting time to now
     */
    override fun init() { startTime = System.nanoTime()}

    /**
     * @brief Checks whether the allowed time has passed since init
     *
     * @TODO: + timeLimit * 0.01? What is the justification or reasoning behind this?
     */
    override fun reachedTermination() = (System.nanoTime() - startTime + epsilon + timeLimit * 0.01) > timeLimit
}
