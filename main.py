import pygame
from pygame.event import get
from pygame.locals import *
from pygame.sprite import Group
import sys
from player import Player
from ball import Ball
from text import Text
from config import Config, rewards_func, new_rewards_func
import random
import time
from DQN import AgentsDQN
from Qlearning import AgentsQT


pygame.init()
conf = Config()
N = 2 # number of player
screen = pygame.display.set_mode(conf.size)
screen_rect = screen.get_rect()
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load(conf.background_image).convert()
players = Group()
ball = Ball(screen_rect.centerx, screen_rect.centery)
agents = []

# TODO: change into multiple players mode
def getGameState(pid, players, ball):
    ret_state = [0, 0, 0, 0, 0, 0]
    for p in players:
        if p.id == pid:
            ret_state[0] = p.rect.centerx
            ret_state[1] = p.rect.centery
        else:
            ret_state[2] = p.rect.centerx
            ret_state[3] = p.rect.centery
    ret_state[4] = ball.rect.centerx
    ret_state[5] = ball.rect.centery
    return ret_state


def initialize_game():
    for i in range(1, N+1):
        team_now = 0
        image = conf.player_image_blue
        if i > N / 2:
            team_now = 1
            image = conf.player_image_red
        pos = conf.init_pos[N][i-1]
        print(int(screen_rect.centerx * pos[0]),
                   int(screen_rect.centery * pos[1]))
        p = Player(team_now, int(screen_rect.centerx * pos[0]),
                   int(screen_rect.centery * pos[1]), i, image)
        players.add(p)
        print("player: id={}, team={}".format(p.id, p.team))
    
def initialize_AI():
    for p in players.sprites():
        #agent = AgentsQT(p.id, N)
        agent = AgentsDQN(p.id, N)
        agents.append(agent)


def reset():
    ball.rect.centerx = screen_rect.centerx
    ball.rect.centery = screen_rect.centery
    ball.if_caught = False
    ball.catcher = -1
    ball.v.x = 0
    ball.v.y = 0
    for p in players.sprites():
        p.rect.centerx = screen_rect.centerx * conf.init_pos[N][p.id-1][0]
        p.rect.centery = screen_rect.centery * conf.init_pos[N][p.id-1][1]
        p.v.x = 0
        p.v.y = 0


# ai interact with game from here
def get_input(pid):
    #        W  S  A  D  Space
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
            print("p-{} shoot, dir in ({},{}) {}, input={}".format(p.id,
                                                                   ball.v.x, ball.v.y, p.shoot_dir, input_array))
            return True
        p.shoot_dir = 99
    return False


def deal_collision():
    stealer_list = []
    holder = None
    stealer = None
    for p in players.sprites():  # check if anyone want to steal the ball
        if pygame.sprite.collide_rect(ball, p):
            if ball.belong(p.id):
                holder = p
            elif p.check_shoot_cd():
                stealer_list += [p]
    if len(stealer_list) == 1:
        stealer = stealer_list[0]
    elif len(stealer_list) > 1:
        stealer = stealer_list[random.randint(0, len(stealer_list) - 1)]

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
    assert N in conf.available_player_numbers

    render_mode = True
    episodes = 2
    FPS = 100

    game_on = True
    score = [0, 0]
    # p1_id = 1
    game_timer = pygame.time.Clock()
    game_time = conf.max_time
    game_timer.tick(FPS)
    initialize_game()
    initialize_AI()
    info = Text()

    # While loop for main logic of the game
    for episode in range(episodes):
        print("episode: ", episode)
        reset()
        game_time = conf.max_time
        game_on = True
        for agent in agents:
            agent.set_state(getGameState(agent.id, players, ball))
            agent.update_greedy()
        step = 0
        # state = []
        # state.append(agents[0].get_state(getGameState(1, players, ball)))
        # state.append(agents[1].get_state(getGameState(2, players, ball)))
        # agents[0].update_greedy()
        # agents[1].update_greedy()
        # print(agents[0].greedy)

        while game_on:
            prev_pos_x = [0 for _i in range(N+1)]
            prev_pos_y = [0 for _i in range(N+1)]
            new_pos_x = [0 for _i in range(N+1)]
            new_pos_y = [0 for _i in range(N+1)]
            # next_state = []
            rewards = [0, 0]
            action = [0 for _i in range(N)]

            # end game with user input
            # for event in pygame.event.get():
            #     if event.type == QUIT:
            #         pygame.quit()
            #         sys.exit()
            #         game_on = False

            # update position
            prev_pos_x[N] = ball.rect.centerx
            prev_pos_y[N] = ball.rect.centery
            for p in players.sprites():
                #input_array = get_input(p.id)
                prev_pos_x[p.id - 1] = p.rect.centerx
                prev_pos_y[p.id - 1] = p.rect.centery
                action[p.id - 1] = agents[p.id - 1].make_decision()
                input_array = get_input_ai(p.id, action[p.id - 1])
                # deal with input & calculate reward
                if deal_player_input(p, ball, input_array):
                    rewards[p.team] -= 1000

            # deal with collision
            if_ball_free, stealer = deal_collision()
            # calculate reward
            if stealer is not None:  # steal the ball
                if if_ball_free:  # if ball is free
                    rewards[stealer.team] += 1500
                else: # if ball is stolen
                    if stealer.team == 0:
                        rewards[0] += 2000
                        rewards[1] -= 2000
                    else:
                        rewards[1] += 2000
                        rewards[0] -= 2000

            ball.update_pos()
            new_pos_x[N] = ball.rect.centerx
            new_pos_y[N] = ball.rect.centery
            shot = ball.in_door()
            if shot >= 0:
                score[shot] += 1
                rewards[shot] += 100000
                rewards[shot-1] -= 10000
                reset()

            # next_state.append(agents[0].get_state(getGameState(1, players, ball)))
            # next_state.append(agents[1].get_state(getGameState(2, players, ball)))

            # agents[0].update_q_table(state[0], action[0], next_state[0], rewards[0])
            # agents[1].update_q_table(state[1], action[1], next_state[1], rewards[1])
            for player in players.sprites():
                new_pos_x[player.id - 1] = player.rect.centerx
                new_pos_y[player.id - 1] = player.rect.centery
            rewards = rewards_func(rewards, prev_pos_x, prev_pos_y, new_pos_x, new_pos_y, N)
            # rewards = new_rewards_func(rewards, prev_pos_x, prev_pos_y, new_pos_x, new_pos_y, N)
            # rewards = [0, 0]
            # print(rewards)


            # Q-learning version
            # for agent in agents:
            #     team_now = 0
            #     if agent.id > N/2:
            #         team_now = 1
            #     agent_state = getGameState(agent.id, players, ball)
            #     # agent_state = [0, 0, 0, 0, 0 ,0]
            #     agent.update(action[agent.id - 1], agent_state, rewards[team_now])

            # DQN version
            for agent in agents:
                team_now = 0
                if agent.id > N/2:
                    team_now = 1
                agent_state = getGameState(agent.id, players, ball)
                agent.store_transition(action[agent.id - 1], rewards[team_now], agent_state)
                # agent_state = [0, 0, 0, 0, 0 ,0]
                if (step > 200) and (step % 5 == 0):
                    print("update when step=", step)
                    agent.update()

            if render_mode:
                screen.blit(background, (0, 0))
                for player in players.sprites():
                    player.render(screen)
                ball.render(screen)
                info.render(screen, score, game_time)
                pygame.display.update()

            game_time -= game_timer.tick(FPS)
            if game_time < 0:
                game_on = False
            step += 1

    # wait for game exit
    # screen.blit(background, (0, 0))
    # for player in players.sprites():
    #     player.render(screen)
    # ball.render(screen)
    # info.render(screen, score, 0)
    # pygame.display.update()

    for agent in agents:
        agent.save_model()
        #agent.plot_cost()

    time.sleep(10)
    pygame.quit()
    sys.exit()
