import numpy as np
import math
from config import Config

conf = Config()

class Agent():

    def __init__(self, id):

        # create Q table
        # The sturcture of Q table:
        #   player's position (9, 9)
        #   opponent's position (8, 8)
        #   ball's position (8, 8)
        #   moving direction
        #   action 0->nothing 1->kick?
        self.id = id
        self.q_table = np.zeros((9, 9, 7, 7, 7, 7, 9, 2))
        # learning rate
        self.alpha = 0.7
        # 
        self.gamma = 0.7
        #
        self.greedy = 0.7

    # to simplify the state of current game
    def get_state(self, state):
        """
        :param state: state is a dictionary containing the current state
        """
        return_state = np.zeros((6,), dtype=int);
        player_x = state['player_x']
        player_y = state['player_y']
        opponent_x = state['opponent_x']
        opponent_y = state['opponent_y']
        ball_x = state['ball_x']
        ball_y = state['ball_y']

        interval_x = 1/12 * conf.width
        return_state[0] = (player_x - (0.125 * conf.width)) // interval_x
        interval_y = 1/12 * conf.height
        return_state[1] = (player_y - (0.125 * conf.height)) // interval_y

        diff_x = opponent_x - player_x
        diff_y = opponent_y - player_y
        if diff_x > 0:
            return_state[2] = math.ceil(math.log10(abs(diff_x) + 1)) + 3
        elif diff_x == 0:
            return_state[2] = 3
        else:
            return_state[2] = 3 - math.ceil(math.log10(abs(diff_x) + 1))
        if diff_y > 0:
            return_state[3] = math.ceil(math.log10(abs(diff_x) + 1)) + 3
        elif diff_y == 0:
            return_state[3] = 3
        else:
            return_state[3] = 3 - math.ceil(math.log10(abs(diff_x) + 1))

        diff_ball_x = ball_x - player_x
        diff_ball_y = ball_y - player_y
        if diff_ball_x > 0:
            return_state[4] = math.ceil(math.log10(abs(diff_x) + 1)) + 3
        elif diff_ball_x == 0:
            return_state[4] = 3
        else:
            return_state[4] = 3 - math.ceil(math.log10(abs(diff_x) + 1))
        if diff_ball_y > 0:
            return_state[5] = math.ceil(math.log10(abs(diff_x) + 1)) + 3
        elif diff_ball_y == 0:
            return_state[5] = 3
        else:
            return_state[5] = 3 - math.ceil(math.log10(abs(diff_x) + 1))

        return return_state

    def update_q_table(self, old_state, current_action, next_state, r):
        next_max_value = np.max(self.q_table[next_state[0], next_state[1], next_state[2],
            next_state[3], next_state[4], next_state[5]])
        self.q_table[old_state[0], old_state[1], old_state[2], next_state[3], next_state[4], 
            next_state[5], current_action] = (1 - self.alpha) * self.q_table[
            old_state[0], old_state[1], old_state[2], next_state[3], next_state[4], 
            next_state[5], current_action] + self.alpha * (r + next_max_value)

    def make_decision(self, state, random=True):
        act = []
        ret_act = 0
        act[0] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 0, 0]
        act[1] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 1, 0]
        act[2] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 2, 0]
        act[3] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 3, 0]
        act[4] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 4, 0]
        act[5] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 5, 0]
        act[6] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 6, 0]
        act[7] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 7, 0]
        act[8] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 8, 0]

        act[9] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 0, 1]
        act[10] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 1, 1]
        act[11] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 2, 1]
        act[12] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 3, 1]
        act[13] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 4, 1]
        act[14] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 5, 1]
        act[15] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 6, 1]
        act[16] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 7, 1]
        act[17] = self.q_table[state[0], state[1], state[2], state[3], state[4], state[5], 8, 1]
        
        if (random):
            if np.random.rand(1) < self.greedy:
                ret_act = np.random.choice(range(18))
            else:
                ret_act = act.index(max(act))
        else:
            ret_act = act.index(max(act))

        if ret_act >= 9:
            return [ret_act - 9, 1]
        else:
            return [ret_act, 0]
        
    def update_greedy(self):
        self.greedy *= 0.95

    def act(self, p, action):
        """
        :param p: sending action to game
        """
        pass

    
