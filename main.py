import pygame
from pygame.locals import *
from pygame.sprite import Group
import sys
import player
from ball import Ball
from config import Config

pygame.init()

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


def collide(ball, player):
    if not(player.check_shoot_cd()):
        return False
    result = pygame.sprite.collide_rect(ball, player)
    return result


def handle_event(p, event): # p = player
    if event.type == KEYDOWN:
        if event.key == K_LEFT:
            p.v.x = -conf.player_v
        elif event.key == K_RIGHT:
            p.v.x = conf.player_v
        if event.key == K_UP:
            p.v.y = -conf.player_v
        elif event.key == K_DOWN:
            p.v.y = conf.player_v

        if event.key == K_SPACE:
            p.shoot(ball, conf.player_power)

    if event.type == KEYUP:
        if event.key == K_LEFT or event.key == K_RIGHT:
            p.v.x = 0
        if event.key == K_UP or event.key == K_DOWN:
            p.v.y = 0



# While loop for main logic of the game
while 1:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    """
    #original method
        for player in players.sprites():
            handle_event(player, event)
            if not(player.catching) and not(ball.if_catched) and collide(ball, player):
                player.catch_ball(ball)
    """

    #Add by Louis
    pressed_keys = pygame.key.get_pressed()
    for player in players.sprites():
        player.inputHandler(pressed_keys, ball)

    catched_player = pygame.sprite.spritecollideany(ball, players)
    if catched_player and not(ball.if_catched):
        if catched_player.check_shoot_cd():
            catched_player.catch_ball(ball)
    #

    screen.blit(background, (0, 0))
    for player in players.sprites():
        player.render(screen)
        if player.catching:
            ball.catched(player)
    ball.render(screen)

    pygame.display.update()

#TODO: Boundary checking for ball
#TODO: Add a input handler for 2 players game