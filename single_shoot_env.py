import pygame
from pygame.event import get
from pygame.locals import *
from pygame.sprite import Group
import sys
from sys import argv
import numpy as np
from numpy import random as np_random
from player import Player
from ball import Ball
from text import Text
from config import Config, single_rewards_func
import time
from DQN import AgentsDQN
from Qlearning import AgentsQT
from DDQN import AgentsDDQN


pygame.init()
conf = Config()
screen = pygame.display.set_mode(conf.size)
screen_rect = screen.get_rect()
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load(conf.background_image).convert()
ball = Ball(screen_rect.centerx, screen_rect.centery)

# TODO: change into multiple players mode
def getGameState_single(p, ball):
    ret_state = [0, 0, 0, 0, 0]
    if ball.belong(p.id):
        ret_state[0] = 1
    ret_state[1] = p.rect.centerx
    ret_state[2] = p.rect.centery
    ret_state[3] = ball.rect.centerx
    ret_state[4] = ball.rect.centery
    return ret_state


def initialize_player():
    team_now = 0
    image = conf.player_image_blue
    # shoot right door
    pos = conf.init_pos[2][0]
    print(int(screen_rect.centerx * pos[0]),
        int(screen_rect.centery * pos[1]))
    player = Player(team_now, int(screen_rect.centerx * pos[0]),
        int(screen_rect.centery * pos[1]), 1, image)
    print("player: id={}, team={}".format(player.id, player.team))
    return player


def initialize_agent(pid, agent_mode):
    agent = None
    if agent_mode == "QT":
        agent = AgentsQT(pid, 1, 5)
    elif agent_mode == "DDQN":
        agent = AgentsDDQN(pid, 1, 5)
    else:
        agent = AgentsDQN(pid, 1, 5)
    return agent


def reset_random(ball, player):
    ball.rect.centerx = screen_rect.centerx + (np_random.rand() - 0.5) * conf.width * 0.65
    ball.rect.centery = screen_rect.centery + (np_random.rand() - 0.5) * conf.height * 0.65
    ball.if_caught = False
    ball.catcher = -1
    ball.v.x = 0
    ball.v.y = 0
    player.rect.centerx = screen_rect.centerx + (np_random.rand() - 0.5) * conf.width * 0.65
    player.rect.centery = screen_rect.centery + (np_random.rand() - 0.5) * conf.height * 0.65
    player.v.x = 0
    player.v.y = 0


# ai interact with game from here
def get_input(pid):
    #              W  S  A  D  Space
    input_array = [0, 0, 0, 0, 0]
    if pid == p1_id and pid > 0:  # deal with user_keyboard input
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_w]:
            input_array[0] = 1
        if pressed_keys[K_s]:
            input_array[1] = 1
        if pressed_keys[K_a]:
            input_array[2] = 1
        if pressed_keys[K_d]:
            input_array[3] = 1
        if pressed_keys[K_SPACE]:
            input_array[4] = 1
        return input_array
    else:
        '''
        for i in range(5):
            if random.random() > 0.5:
                input_array[i] = 1
        '''
        return input_array


def get_input_ai(pid, action):
    ret_array = [0, 0, 0, 0, 0]
    if action[0] == 1 or action[0] == 2 or action[0] == 8:
        ret_array[0] = 1
    if action[0] == 2 or action[0] == 3 or action[0] == 4:
        ret_array[3] = 1
    if action[0] == 4 or action[0] == 5 or action[0] == 6:
        ret_array[1] = 1
    if action[0] == 6 or action[0] == 7 or action[0] == 8:
        ret_array[2] = 1
    if action[1] == 1:
        ret_array[4] = 1
    return ret_array


def deal_player_input(p, ball, input_array):
    p.input_handler(input_array)
    if p.shoot_dir < 99:
        if ball.belong(p.id):
            p.shoot_update()
            ball.shoot_ball(p.shoot_dir)
            # rewards[p.id - 1] -= 1000
            print("p-{} shoot, dir in ({},{}) {}, input={}".format(p.id, ball.v.x, ball.v.y, p.shoot_dir, input_array))
            return ball.v.x
        p.shoot_dir = 99
    return None


def deal_collision_single(p):
    holder = None
    stealer = None
    if pygame.sprite.collide_rect(ball, p):
        if ball.belong(p.id):
            holder = p
        elif p.check_shoot_cd():
            stealer = p
    if stealer is None and holder is not None:  # still hold the ball
        ball.copy_pos(holder.rect.centerx, holder.rect.centery)
    elif stealer is not None:  # steal the ball
        if ball.belong(-1):  # if ball is free
            ball.caught(stealer.id)
            return True, stealer
        elif ball.check_time_up():  # if ball is stolen
            ball.caught(stealer.id)
            return False, stealer
    return True, None


if __name__ == "__main__":

    render_mode = True
    episodes = 100
    FPS = 500
    game_on = True
    score = 0
    score_history = []

    game_timer = pygame.time.Clock()
    game_timer.tick(FPS)
    agent_mode = 'QT'
    player = initialize_player()
    agent = initialize_agent(player.id, agent_mode)
    info = Text()
    step = 0
    test_mode = False
    if len(argv) > 1:
        agent_mode = argv[1]
        if len(argv) > 2 and argv[2] == 'test':
            test_mode = True
    print("test_mode:", test_mode)

    # While loop for main logic of the game
    for episode in range(episodes):
        print("episode: ", episode)
        reset_random(ball, player)
        game_time = 10000 # conf.max_time
        game_on = True
        agent.set_state(getGameState_single(player, ball))
        agent.update_greedy()
        prev_pos_x = [player.rect.centerx, ball.rect.centerx]
        prev_pos_y = [player.rect.centery, ball.rect.centery]

        while game_on:
            
            new_pos_x = [0, 0]
            new_pos_y = [0, 0]
            # next_state = []
            reward = 0
            action = [-1, -1]

            if test_mode:
                action = agent.make_decision(no_random = True)
            elif step < 300 and not(agent.has_model) and agent_mode != 'QT':
                action = agent.make_random_decision()
            else:
                action = agent.make_decision()
            input_array = get_input_ai(player.id, action)
            # deal with input & calculate reward
            # if deal_player_input(player, ball, input_array):
            #     rewards -= 1000
            ball_v_x = deal_player_input(player, ball, input_array)
            if ball_v_x is not None:
                if (ball_v_x > 0 and player.team == 0) or (ball_v_x < 0 and player.team == 1):
                    reward += 1000
                    print("good shoot")
                if (ball_v_x < 0 and player.team == 0) or (ball_v_x > 0 and player.team == 1):
                    reward -= 1000
                    print("bad shoot")

            # deal with collision
            _, stealer = deal_collision_single(player)
            # calculate reward
            if stealer is not None:  # steal the ball
                reward += 2000
                print("catch ball")

            ball.update_pos()
            
            shot = ball.in_door()
            if shot == player.team:
                score += 1
                reward += 100000
                print('get a score!!!!')
                score_history.append(score)
                game_on = False
            elif shot >= 0:
                reward -= 5000
                print('in the wrong door!!!')
                score_history.append(score)
                game_on = False
            
            if render_mode:
                screen.blit(background, (0, 0))
                player.render(screen)
                ball.render(screen)
                info.render(screen, score, game_time)
                pygame.display.update()

            new_pos_x[1] = ball.rect.centerx
            new_pos_y[1] = ball.rect.centery
            new_pos_x[0] = player.rect.centerx
            new_pos_y[0] = player.rect.centery
            # rewards = rewards_func(rewards, prev_pos_x, prev_pos_y, new_pos_x, new_pos_y, N)
            reward += single_rewards_func(prev_pos_x, prev_pos_y, new_pos_x, new_pos_y, player.team)
            #print(reward)

            agent_state = getGameState_single(player, ball)
            if agent_mode == "QT":
                # Q-learning version
                agent.update(action, agent_state, reward)
            else:
                # DQN version
                agent.store_transition(action, reward, agent_state)
                if (step > 1000) and (step % 10 == 0) and not(test_mode):
                    agent.update()

            game_time -= game_timer.tick(FPS)
            if game_time < 0:
                game_on = False
                score_history.append(score)
            step += 1
            prev_pos_x = new_pos_x
            prev_pos_y = new_pos_y

        # if not(test_mode) and episode % 100 == 0:
        #     agent.save_model(postfix="-"+str(episode))

    # if not(test_mode):
    #     agent.save_model(postfix = "-after_test")
    #     agent.plot_qvalue()
    #     agent.plot_reward()

    import matplotlib.pyplot as plt
    plt.plot(np.array(score_history))
    plt.ylabel('score_history')
    plt.xlabel('training episode')
    plt.grid()
    plt.savefig(agent.path+"score_history.jpg")
    plt.show()

    time.sleep(10)
    pygame.quit()
    sys.exit()
