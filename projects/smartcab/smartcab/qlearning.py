import random

class Q:


    def __init__(self, actions, rate=.01, epsilon=.9, alpha=.5, gamma=.2):
        self._Q = dict()
        self.INIT_VAL = 200
        self.ALPHA = alpha            # Learning Rate
        self.EPSILON = epsilon        # Randomness Prob
        self.GAMMA = gamma            # Discount factor
        self.actions = actions        #[None, 'forward', 'left', 'right']
        self.rate = rate;

    def update_epsilon(self):
        self.EPSILON -= self.rate if self.EPSILON >= 0.03 else 0
        print "EPSILON {}".format(self.EPSILON)

    def update_gamma(self):
        self.GAMMA += self.rate if self.GAMMA >= 1 else 0
        self.GAMMA = self.GAMMA

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
