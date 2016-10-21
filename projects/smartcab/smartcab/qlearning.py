import random

class Q:


    def __init__(self, actions):
        self._Q = dict()
        self.EPSILON = 0.2 # Randomness Prob
        self.ALPHA = 0.5 # Learning Rate
        self.GAMMA = 0.5 # Discount factor
        self.INIT_VAL = 200
        self.actions = actions #[None, 'forward', 'left', 'right']

    def _update_alpha(self):
        self.ALPHA = self.ALPHA/2

    def _update_epsilon(self):
        self.EPSILON = self.EPSILON/2

    def _get_Q(self, state, action):

        pair = (state, action)
        if pair not in self._Q:
            self._Q[pair] = self.INIT_VAL
        return self._Q[pair]

    def update_Q(self, s, a, r, s_x):
        s = frozenset(s.items())
        s_x = frozenset(s_x.items())

        pair = (s, a)
        state_ut =  (r + self.GAMMA * self.max_Q_and_action(s_x)[0])
        self._Q[pair] =  (1-self.ALPHA) * self._Q[pair] + self.ALPHA * state_ut

    def policy(self, state):

        state = frozenset(state.items())
        value = random.random
        if value < self.EPSILON:
            return random.choice(self.actions)
        else:
            return self.max_Q_and_action(state)[1]

    def max_Q_and_action(self, state):
        argmax = None
        max_q = None
        for action in self.actions:
            current_q = self._get_Q(state,action)
            if(max_q == None or current_q > max_q):
                max_q = current_q
                argmax = action
        return (max_q, argmax)
