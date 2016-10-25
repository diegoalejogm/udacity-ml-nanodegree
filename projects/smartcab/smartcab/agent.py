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
        # Preformance Metrics
        self.performance = PerformanceMetrics()
        self.iter = 0
        self.score = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.iter+=1


    def _arrived_at_dest(self):
        return  self.planner.next_waypoint() == None

    def _generate_state(self, inputs):

        state = dict(inputs)
        state.pop('left', None)
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

        #
        # Data Performance Metrics
        #

        self.performance.score += reward
        if self.iter > 90:
            self.performance.score_last_10 += reward

        if self._arrived_at_dest():
            self.Q.update_epsilon()
            self.performance.successes += 1

            if self.Q.EPSILON <= .03 and self.performance.min_epsilon_iter == 101:
                self.performance.min_epsilon_iter = self.iter
            if self.iter > 90:
                self.performance.last_10_successes += 1
                new_avg_deadline_last_10 = self.performance.avg_perc_to_deadline_last_10 * (self.iter-91) + perc_to_deadline
                self.performance.avg_perc_to_deadline_last_10 = new_avg_deadline_last_10 / (self.iter-90)


        if self._arrived_at_dest() or deadline == 0:

            perc_to_deadline = float(t)/(t+deadline) if deadline > 0 else 3
            new_avg_deadline = self.performance.avg_perc_to_deadline * (self.iter-1) + perc_to_deadline

            self.performance.avg_perc_to_deadline = new_avg_deadline / self.iter
            if self.iter > 90:
                new_avg_deadline_last_10 = self.performance.avg_perc_to_deadline_last_10 * (self.iter-91) + perc_to_deadline
                self.performance.avg_perc_to_deadline_last_10 = new_avg_deadline_last_10 / (self.iter-90)


        # self.Q.update_gamma() if self.planner.next_waypoint() == None
#        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def test():

    d = dict()
    index = []
    d['alpha'] = []
    d['gamma'] = []
    d['epsilon'] = []
    d['avg_perc_to_deadline'] = []
    d['avg_perc_to_deadline_last_10'] = []
    d['eps_iters'] = []
    d['last_10_successes'] = []
    d['scores'] = []
    d['scores_last_10'] = []
    d['successes'] = []

    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for alpha_i in values:
        for gamma_i in values:
            for epsilon_i in values:

                eps_iters = []
                successes = []
                last_10_successes = []
                scores = []
                scores_last_10 = []
                avg_perc_to_deadline = []
                avg_perc_to_deadline_last_10 = []

                for i in range(20):
                    """Run the agent for a finite number of trials."""

                    # Set up environment and agent
                    e = Environment(num_dummies=3)  # create environment (also adds some dummy traffic)
                    Q_learner = Q(actions=e.valid_actions, alpha=alpha_i, gamma=gamma_i, epsilon=epsilon_i)
                    a = e.create_agent(LearningAgent, Q=Q_learner)  # create agent
                    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                    # Now simulate it
                    sim = Simulator(e, update_delay=0.000000000000000005, display=False)  # create simulator (uses pygame when display=True, if available)
                    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                    sim.run(n_trials=100)  # run for a specified number of trials

                    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                    avg_perc_to_deadline.append(a.performance.avg_perc_to_deadline)
                    avg_perc_to_deadline_last_10.append(a.performance.avg_perc_to_deadline_last_10)
                    eps_iters.append(a.performance.min_epsilon_iter)
                    scores.append(a.performance.score)
                    last_10_successes.append(a.performance.last_10_successes)
                    successes.append(a.performance.successes)
                    scores_last_10.append(a.performance.score_last_10)

                d['alpha'].append(alpha_i)
                d['gamma'].append(gamma_i)
                d['epsilon'].append(epsilon_i)
                d['avg_perc_to_deadline'].append(np.average(avg_perc_to_deadline))
                d['avg_perc_to_deadline_last_10'].append(np.average(avg_perc_to_deadline_last_10))
                d['eps_iters'].append(np.average(eps_iters))
                d['last_10_successes'].append(np.average(last_10_successes))
                d['scores'].append(np.average(scores))
                d['scores_last_10'].append(np.average(scores_last_10))
                d['successes'].append(np.average(successes))
                #index.append("A:{} E:{} G:{}".format(alpha_i,epsilon_i,gamma_i))

    df = pd.DataFrame(d)
    df.to_csv("results3.csv", sep='\t')


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=3)  # create environment (also adds some dummy traffic)
    Q_learner = Q(actions=e.valid_actions, alpha=1, gamma=0, epsilon=0)
    a = e.create_agent(LearningAgent, Q=Q_learner)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.2, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials

    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

#index.append("A:{} E:{} G:{}".format(alpha_i,epsilon_i,gamma_i))


if __name__ == '__main__':
    run()
