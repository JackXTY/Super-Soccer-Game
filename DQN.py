import numpy as np
import math
from config import Config
import os

from collections import deque
from tensorflow.keras import models, layers, optimizers

conf = Config()


class Agent():
    def __init__(self, id, game_mode, model_root_path="./model/", train=True):
        self.id = id
        self.type = "Agent"
        self.game_mode = game_mode
        self.model_path = model_root_path + self.type + \
            "/" + game_mode + "/" + str(id) + ".model"
        self.model = self.load_model()

    def set_state(self, state):
        pass

    def get_state(self, state):
        pass

    def update(self):
        pass

    def make_decision(self, random):
        pass

    def reset(self):
        pass

    def load_model(self):
        pass

    def update_greedy(self):
        pass

    def save_model(self):
        pass


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
        self.state = state

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

    def update(self, current_action, next_state, r):
        old_state = self.state
        next_max_value = np.max(self.q_table[next_state[0], next_state[1], next_state[2],
                                             next_state[3], next_state[4], next_state[5]])
        self.q_table[old_state[0], old_state[1], old_state[2], next_state[3], next_state[4],
                     next_state[5], current_action] = (1 - self.alpha) * self.q_table[
            old_state[0], old_state[1], old_state[2], next_state[3], next_state[4],
            next_state[5], current_action] + self.alpha * (r + self.gamma * next_max_value)

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

# TODO: ENV!


class AgentsDDQN(Agent):
    def __init__(self, action_set):
        self.gamma = 1
        self.model = self.init_netWork()
        self.batch_size = 128
        self.memory = deque(maxlen=2000000)
        self.greedy = 1
        self.action_set = action_set

    def get_state(self, state):
        """
        提取游戏state中我们需要的数据
        :param state: 游戏state
        :return: 返回提取好的数据
        """
        return_state = np.zeros((6,))
        return_state[0] = state['player_x']
        return_state[1] = state['player_y']
        return_state[2] = state['opponent_x']
        return_state[3] = state['opponent_y']
        return_state[4] = state['ball_x']
        return_state[5] = state['ball_y']

        return return_state

    def init_netWork(self):
        """
        构建模型
        :return:
        """
        model = models.Sequential([
            layers.Dense(64 * 4, activation="tanh",
                         input_dim=self.observation_space.shape[0]),
            layers.Dense(18, activation="linear")
        ])

        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def add_memory(self, sample):
        self.memory.append(sample)

    def update_greedy(self):
        # 小于最小探索率的时候就不进行更新了。
        if self.greedy > 0.01:
            self.greedy *= 0.995
