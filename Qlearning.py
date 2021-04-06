import numpy as np
import math
import pandas as pd
from config import Config
import os

import tensorflow as tf
from tensorflow.compat.v1 import placeholder, variable_scope, GraphKeys, get_variable, squared_difference, Session, get_collection, assign, global_variables_initializer, train
from collections import deque
from tensorflow.keras import models, layers, optimizers
from agent import Agent

conf = Config()

class AgentsQT(Agent):
    def __init__(self, id, N):
        # create Q table
        # The sturcture of Q table:
        #   player's position (9, 9)
        #   opponent's relative position (7, 7)
        #   ball's relative position (7, 7)
        #   moving direction
        #   action 0->nothing 1->kick?
        
        self.id = id
        self.path = "./model/" + str(N) + "/" + str(id) + ".npy"
        self.state = []
        self.next_state = []
        self.has_model = os.path.exists(self.path)
        if self.has_model:
            self.greedy = 0.005
            self.q_table = self.load_model()
        else:
            self.q_table = np.zeros((9, 9, 7, 7, 7, 7, 9, 2))
            # exploration strategy
            self.greedy = 0.9

        # learning rate
        self.alpha = 1
        # discount factor
        self.gamma = 0.7

    def set_state(self, state):
        self.state = self.get_state(state)

    # to simplify the state of current game
    def get_state(self, state):
        return_state = np.zeros((6,), dtype=int)
        player_x = state[0]
        player_y = state[1]
        opponent_x = state[2]
        opponent_y = state[3]
        ball_x = state[4]
        ball_y = state[5]

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

    def update(self, current_action, game_state, r):
        next_state = self.get_state(game_state)
        old_state = self.state
        next_max_value = np.max(self.q_table[next_state[0], next_state[1], next_state[2],
                                             next_state[3], next_state[4], next_state[5]])
        self.q_table[old_state[0], old_state[1], old_state[2], next_state[3], next_state[4],
                     next_state[5], current_action] = (1 - self.alpha) * self.q_table[
            old_state[0], old_state[1], old_state[2], next_state[3], next_state[4],
            next_state[5], current_action] + self.alpha * (r + self.gamma * next_max_value)
        self.state = next_state

    def make_decision(self, random=True):
        state = self.state
        act = []
        ret_act = 0
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 0, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 1, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 2, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 3, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 4, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 5, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 6, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 7, 0])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 8, 0])

        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 0, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 1, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 2, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 3, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 4, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 5, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 6, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 7, 1])
        act.append(self.q_table[state[0], state[1], state[2],
                                state[3], state[4], state[5], 8, 1])

        if (random):
            if np.random.rand(1) < self.greedy:
                ret_act = np.random.choice(range(18))
            else:
                #ret_act = act.index(max(act))
                max_val = max(act)
                ret_acts = []
                for i in range(18):
                    if act[i] == max_val:
                        ret_acts.append(i)
                ret_act = np.random.choice(ret_acts)
        else:
            ret_act = act.index(max(act))

        if ret_act >= 9:
            return [ret_act - 9, 1]
        else:
            return [ret_act, 0]

    def update_greedy(self):
        self.greedy *= 0.95

    def load_model(self):
        model = np.load(self.path)
        return model

    def save_model(self):
        np.save(self.path, self.q_table)
        print(self.path, " saved")
