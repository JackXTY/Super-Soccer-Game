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

# TODO: ENV!

# refernce from https://github.com/MorvanZhou
class AgentsDQN(Agent):
    def __init__(self, id, N):
        self.id = id
        self.path = "./model/DQN/" + str(N) + "/" + str(id)
        self.state = []
        self.next_state = []
        self.has_model = os.path.exists(self.path)
        # learning rate
        if self.has_model:
            self.greedy = 0.05
        else:
            # exploration strategy
            self.greedy = 0.1
        # discount factor
        self.gamma = 0.3
        # number of features
        self.features = 6
        # number of actions
        self.actions = 16
        self.replace_target_iter = 300
        self.memory_size = 500
        self.epsilon = 0.6
        self.epsilon_max = 1
        self.epsilon_increment = 0.001

        self.step_counter = 0
        self.memory = np.zeros((self.memory_size, self.features*2+2))
        self.build_network()

        self.sess = Session()
        self.batch_size = 16
        # tf.summary.FileWriter("logs/", self.sess.graph)
        print(self.id, self.sess)
        self.saver = train.Saver()

        if not(os.path.exists(self.path)):
            self.sess.run(global_variables_initializer())
        else:
            self.load_model()
        self.cost_history = []
    
    def set_state(self, state):
        self.state = state
        # try to give up get_state in DQN, just use original state
    
    def build_network(self):
        tf.compat.v1.disable_eager_execution()
        # evaluate network
        self.s_eval = placeholder(tf.float32, [None, self.features], name='s')
        self.q_target = placeholder(tf.float32, [None, self.actions], name='Q_target')
        with variable_scope('eval_net' + str(self.id)) as scope:
            #scope.reuse_variables()
            c_names = ['eval_net_params' + str(self.id), GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 50
            w_init = tf.random_normal_initializer(0.01)
            b_init = tf.constant_initializer(0.01)
            # first layer. collections is used later when assign to target net
            with variable_scope('l1'):
                w1 = get_variable('w1', [self.features, n_l1], initializer=w_init, collections=c_names)
                b1 = get_variable('b1', [1, n_l1], initializer=b_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_eval, w1) + b1)
            # second layer. collections is used later when assign to target net
            with variable_scope('l2'):
                w2 = get_variable('w2', [n_l1, self.actions], initializer=w_init, collections=c_names)
                b2 = get_variable('b2', [1, self.actions], initializer=b_init, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
        with variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        with variable_scope('train'):
            #self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
            self._train_op = tf.compat.v1.train.AdagradOptimizer(self.greedy).minimize(self.loss)
        
        # target network
        self.s_target = placeholder(tf.float32, [None, self.features], name='s_')    # input
        with variable_scope('target_net' + str(self.id)):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params' + str(self.id), GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with variable_scope('l1'):
                w1 = get_variable('w1', [self.features, n_l1], initializer=w_init, collections=c_names)
                b1 = get_variable('b1', [1, n_l1], initializer=b_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_target, w1) + b1)

            # second layer. collections is used later when assign to target net
            with variable_scope('l2'):
                w2 = get_variable('w2', [n_l1, self.actions], initializer=w_init, collections=c_names)
                b2 = get_variable('b2', [1, self.actions], initializer=b_init, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, action, reward, state_new):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        action_number = action[0] - 1 + action[1] * 8
        transition = np.hstack((self.state, [action_number, reward], state_new))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        self.state = state_new

    def make_decision(self):
        # observation = observation[np.newaxis, :]
        observation = np.array(self.state).reshape([1, self.features])
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s_eval: observation})
            action = np.argmax(actions_value[0][:])
            action_0 = action % 8 + 1
            action_1 = math.floor(action / 8)
            return [action_0, action_1]
        else:
            action_0 = np.random.randint(1, 9)
            action_1 = np.random.randint(0, 2)
            return [action_0, action_1]
    
    def replace_target_params(self):
        t_params = get_collection('target_net_params' + str(self.id))
        e_params = get_collection('eval_net_params' + str(self.id))
        self.sess.run([assign(t, e) for t, e in zip(t_params, e_params)])
    
    def update(self):

        # check to replace target parameters
        if self.step_counter % self.replace_target_iter == 0:
            self.replace_target_params()
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # print(self.memory)
        # batch_memory = []
        # for si in sample_index:
        #     batch_memory.append([self.memory.iloc[si, :]])
        batch_memory = self.memory[sample_index, :]
        #print(batch_memory)

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_target: batch_memory[:, -self.features:],  # fixed params
                self.s_eval: batch_memory[:, :self.features]  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.features].astype(int)
        reward = batch_memory[:, self.features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s_eval: batch_memory[:, :self.features],
                                                self.q_target: q_target})
        self.cost_history.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step_counter += 1
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def update_greedy(self):
        self.greedy *= 0.95

    def load_model(self):
        self.saver.restore(self.sess, self.path)

    def save_model(self):
        try:
            self.saver.save(self.sess, self.path)
            print(self.path + 'saved successfully')
            self.plot_cost()
        except:
            print('ERROR: can not save the model')


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
