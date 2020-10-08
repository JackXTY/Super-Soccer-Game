import pygame
from pygame.locals import *
from pygame.sprite import Group
import sys
import player
from ball import Ball
from config import Config
import socket


#pygame.init()

conf = Config()

screen = pygame.display.set_mode(conf.size)
screen_rect = screen.get_rect()
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load(conf.background_image).convert()

p1 = player.Player(0, int(screen_rect.centerx/2), screen_rect.centery, conf.player_image_blue)
p2 = player.Player(1, int(screen_rect.centerx*3/2), screen_rect.centery, conf.player_image_red)

players = Group()
players.add(p1)
players.add(p2)

ball = Ball(screen_rect.centerx, screen_rect.centery)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 6666))



# While loop for main logic of the game
while True:
    s.send(("[" + str(p1.rect.centerx) + "/" + str(p1.rect.centery) + "]").encode('utf8'))
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    #Add by Louis
    pressed_keys = pygame.key.get_pressed()
    p1.inputHandler(pressed_keys, ball)

    # this part about catch ball may needed to be dealt in server
    '''
    catched_player = pygame.sprite.spritecollideany(ball, players)
    if catched_player and not(ball.if_catched):
        if catched_player.check_shoot_cd():
            catched_player.catch_ball(ball)
    '''

    #s.send(("["+p1.rect.centerx+"/"+p1.rect.centery+"]").encode('utf8'))

    # need to receive message from server

    screen.blit(background, (0, 0))
    for player in players.sprites():
        player.render(screen)
        if player.catching:
            ball.catched(player)
    ball.render(screen)

    pygame.display.update()

# TODO: Boundary checking for ball
# TODO: Steal ball checking
# TODO: Add a input handler for 2 players game