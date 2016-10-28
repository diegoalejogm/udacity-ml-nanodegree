import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from qlearning import Q
from performance_metrics import PerformanceMetrics
import numpy as np
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, Q):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = Q

        # Preformance Metrics --------
        self.performance = PerformanceMetrics()
        self.iter = 0
        self.score = 0
        #--------

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.iter+=1


    def _arrived_at_dest(self):
        return  self.planner.next_waypoint() == None

    def _generate_state(self, inputs):

        state = dict(inputs)
        state.pop('right', None)
        state['next'] = self.planner.next_waypoint()
        return state

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self._generate_state(inputs)

        # TODO: Select action according to your policy
        action = self.Q.policy(self.state)

        # Execute action and get reward
        reward = self.env.act(self,action)

        # TODO: Learn policy based on state, action, reward
        new_inputs = self.env.sense(self)
        s_x = self._generate_state(new_inputs)
        self.Q.update_Q(self.state, action, reward, s_x)

        if self._arrived_at_dest():
            self.Q.update_epsilon()

        #
        # Data Performance Metrics
        #

        ## SCORES
        self.performance.scores += reward
        if self.iter > 90:
            self.performance.scores_last_10 += reward

        ## TOTAL STEPS
        self.performance.steps += 1   # Add one step to the count
        if self.iter > 90:
            self.performance.steps_last_10 += reward

        ## PENALTIES
        if reward < 0: # Assign penalty if reward is negative
            self.performance.penalties += 1
            if self.iter > 90:
                self.performance.penalties_last_10 += 1

        ## SUCCESSES
        if self._arrived_at_dest():
            self.performance.successes += 1
            if self.iter > 90:
                self.performance.successes_last_10 += 1

        ## % TO DEADLINE
        if self._arrived_at_dest() or deadline == 0:

            perc_to_deadline = float(t)/(t+deadline) if deadline > 0 else 3

            new_avg_deadline = self.performance.perc_to_deadline * (self.iter-1) + perc_to_deadline
            self.performance.perc_to_deadline = new_avg_deadline / self.iter

            if self.iter > 90:
                new_avg_deadline_last_10 = self.performance.perc_to_deadline_last_10 * (self.iter-91) + perc_to_deadline
                self.performance.perc_to_deadline_last_10 = new_avg_deadline_last_10 / (self.iter-90)

        # self.Q.update_gamma() if self.planner.next_waypoint() == None
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def test():

    d = dict()
    index = []
#----
    d['alpha'] = []
    d['gamma'] = []
    d['epsilon'] = []
#----
    d['perc_to_deadline'] = []
    d['penalties'] = []
    d['scores'] = []
    d['steps'] = []
    d['successes'] = []

    d['perc_to_deadline_last_10'] = []
    d['penalties_last_10'] = []
    d['scores_last_10'] = []
    d['steps_last_10'] = []
    d['successes_last_10'] = []
#----

    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    N_TRIALS = 100

    for alpha_i in values:  # Simulate with all combinations
        for gamma_i in values:
            for epsilon_i in values:
                ## Init Arrays
                #----
                perc_to_deadline = []
                penalties = []
                scores = []
                steps = []
                successes = []
                #----
                perc_to_deadline_last_10 = []
                penalties_last_10 = []
                scores_last_10 = []
                steps_last_10 = []
                successes_last_10 = []
                #----

                for i in range(20):
                    """Run the agent for a finite number of trials."""

                    # Set up environment and agent
                    e = Environment(num_dummies=3)  # create environment (also adds some dummy traffic)
                    Q_learner = Q(actions=e.valid_actions, alpha=alpha_i, gamma=gamma_i, epsilon=epsilon_i)
                    a = e.create_agent(LearningAgent, Q=Q_learner)  # create agent
                    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

                    # Now simulate it
                    sim = Simulator(e, update_delay=0.000000000000000000005, display=False)  # create simulator (uses pygame when display=True, if available)

                    sim.run(n_trials=N_TRIALS)  # run for a specified number of trials

                    p = a.performance

                    ## Append performance results
                    #----
                    perc_to_deadline.append(p.perc_to_deadline)
                    penalties.append(p.penalties)
                    scores.append(p.scores)
                    steps.append(p.steps)
                    successes.append(p.successes)
                    #----
                    perc_to_deadline_last_10.append(p.perc_to_deadline_last_10)
                    penalties_last_10.append(p.penalties_last_10)
                    scores_last_10.append(p.scores_last_10)
                    steps_last_10.append(p.steps_last_10)
                    successes_last_10.append(p.successes_last_10)
                    #----

                d['alpha'].append(alpha_i)
                d['gamma'].append(gamma_i)
                d['epsilon'].append(epsilon_i)

                ## Average results for given configuration
                #----
                d['perc_to_deadline'].append(np.average(perc_to_deadline))
                d['penalties'].append(np.average(penalties))
                d['scores'].append(np.average(scores))
                d['steps'].append(np.average(steps))
                d['successes'].append(np.average(successes))
                #----
                d['perc_to_deadline_last_10'].append(np.average(perc_to_deadline_last_10))
                d['penalties_last_10'].append(np.average(penalties_last_10))
                d['scores_last_10'].append(np.average(scores_last_10))
                d['steps_last_10'].append(np.average(steps_last_10))
                d['successes_last_10'].append(np.average(successes_last_10))
                #----


    df = pd.DataFrame(d)
    df.to_csv("results.csv", sep='\t')


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=20)  # create environment (also adds some dummy traffic)
    Q_learner = Q(actions=e.valid_actions, alpha=1, gamma=.4, epsilon=.0)
    a = e.create_agent(LearningAgent, Q=Q_learner)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.2, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1)  # run for a specified number of trials

    df = pd.DataFrame.from_dict(a.Q._Q, orient='index')

    f = open('table.html', 'w')
    f.write(df.to_html())

    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
