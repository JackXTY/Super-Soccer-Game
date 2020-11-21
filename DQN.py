import numpy as np


class Agent():

    def __init__(self, id):

        # create Q table
        self.q_table = np.zeros((6, 6, 6, 2))
        # learning rate
        self.alpha = 0.7
        # 
        self.gamma = 0.7
        #
        self.greedy = 0.7

    # TODO: to simplify the state of current game
    def get_state(self, state):
        pass

    def update_q_table(self, old_state, current_action, next_state, r):
        
        next_max_value = np.max(self.q_table[next_state[0], next_state[1], next_state[2]])
        self.q_table[old_state[0], old_state[1], old_state[2], current_action] = (1 - self.alpha) * self.q_table[
        old_state[0], old_state[1], old_state[2], current_action] + self.alpha * (r + next_max_value)

    def make_decision(self, state, random=True):
        pass

    def update_greedy(self):
        self.greedy *= 0.95

    def act(self, p, action):
        """
        :param p: sending action to game
        """
        pass

    
