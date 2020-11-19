import pygame
from pygame.locals import *
from pygame.sprite import Group
import sys
from player import Player
from ball import Ball
from text import Text
from config import Config
import random
import time

pygame.init()
conf = Config()
N = conf.total_number
screen = pygame.display.set_mode(conf.size)
screen_rect = screen.get_rect()
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load(conf.background_image).convert()
players = Group()
ball = Ball(screen_rect.centerx, screen_rect.centery)


def initialize_game():
    for i in range(1, N + 1):
        team_now = 0
        image = conf.player_image_blue
        if i > N / 2:
            team_now = 1
            image = conf.player_image_red
        pos = conf.init_pos[i]
        p = Player(team_now, int(screen_rect.centerx * pos[0]), int(screen_rect.centery * pos[1]),
                   i, image)
        players.add(p)
        print("player: id={}, team={}".format(p.id, p.team))


def reset():
    ball.rect.centerx = screen_rect.centerx
    ball.rect.centery = screen_rect.centery
    ball.if_caught = False
    ball.catcher = -1
    ball.v.x = 0
    ball.v.y = 0
    for p in players.sprites():
        p.rect.centerx = screen_rect.centerx * conf.init_pos[p.id][0]
        p.rect.centery = screen_rect.centery * conf.init_pos[p.id][1]
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


game_on = True
score = [0, 0]
p1_id = 1
game_timer = pygame.time.Clock()
game_time = conf.max_time
game_timer.tick()
initialize_game()
info = Text()

# While loop for main logic of the game
while game_on:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # deal with input
    for p in players.sprites():
        input_array = get_input(p.id)
        p.input_handler(input_array)
        if p.shoot_dir < 99:
            if ball.belong(p.id):
                p.shoot_update()
                ball.shoot_ball(p.shoot_dir)
                print("p-{} shoot, dir in ({},{}) {}, input={}".format(p.id, ball.v.x, ball.v.y, p.shoot_dir, input_array))
            p.shoot_dir = 99

    # deal with collision
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
        elif ball.check_time_up():  # if ball is stolen
            ball.caught(stealer.id)

    ball.update_pos()
    shot = ball.in_door()
    if shot >= 0:
        score[shot] += 1
        reset()

    screen.blit(background, (0, 0))
    for player in players.sprites():
        player.render(screen)
    ball.render(screen)
    info.render(screen, score, game_time)
    pygame.display.update()

    game_time -= game_timer.tick()
    if game_time < 0:
        game_on = False


# wait for game exit
screen.blit(background, (0, 0))
for player in players.sprites():
    player.render(screen)
ball.render(screen)
info.render(screen, score, 0)
pygame.display.update()

time.sleep(1000)
pygame.quit()
sys.exit()
