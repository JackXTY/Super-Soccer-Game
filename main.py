import pygame
from pygame.locals import *
from pygame.sprite import Group
import sys
from player import Player
from ball import Ball
from config import Config
import random

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
        p = Player(team_now, int(screen_rect.centerx * pos[2]), int(screen_rect.centery * pos[3]),
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


def get_input(pid):  # here may need to adjust with the ai
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
initialize_game()


# While loop for main logic of the game
while game_on:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # deal with input
    for p in players.sprites():
        input_array = get_input(p.id)
        p.input_hadler(input_array)
        if p.shoot_dir < 99:
            if ball.belong(p.id):
                p.shoot_update()
                ball.shoot_ball(p.shoot_dir)
            p.shoot_dir = 99

    # deal with collision
    stealer_list = []
    for p in players.sprites():  # check if anyone want to steal the ball
        if pygame.sprite.collide_rect(ball, p):
            stealer_list += [p]
    controller = None
    if len(stealer_list) == 1:
        controller = stealer_list[0]
    elif len(stealer_list) > 1:
        controller = stealer_list[random.randint(0, len(stealer_list) - 1)]
    if controller is not None:
        if ball.belong(-1):  # if ball is free
            ball.caught(controller.id)
        elif ball.belong(controller):  # if ball is caught by original catcher
            ball.copy_pos(controller.rect.centerx, controller.rect.centery)
        elif ball.check_time_up():  # if ball is stolen
            ball.caught(controller)

    ball.update_pos()
    shot = ball.in_door()
    if shot >= 0:
        score[shot] += 1
        reset()

    screen.blit(background, (0, 0))
    for player in players.sprites():
        player.render(screen)
    ball.render(screen)

    pygame.display.update()

# wait for game exit
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
