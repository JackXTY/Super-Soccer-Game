import pygame
from pygame.locals import *
from pygame.sprite import Group
import sys
import player
from ball import Ball
from config import Config, compress, decompress
import socket
import select

pygame.init()

conf = Config()

screen = pygame.display.set_mode(conf.size)
screen_rect = screen.get_rect()
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load(conf.background_image).convert()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 6666))

while True:
    pre_data = s.recv(2048).decode('utf8')
    if pre_data != None:
        break
pre_data = decompress(pre_data)
team_now = 0
if pre_data[0] > conf.total_number/2:
    team_now = 1
print("my_player: id={}, team={}".format(pre_data[0], team_now))
p1 = player.Player(0, int(screen_rect.centerx*pre_data[1]), int(screen_rect.centery*pre_data[2]),
                   pre_data[0], conf.player_image_blue)

# read other player data directly from config
other_players = Group()
for i in range(1, conf.total_number+1):
    if i == p1.id:
        continue
    team_now = 0
    image = conf.player_image_blue
    print("other_player: id={}, team={}".format(i, team_now))
    if i > conf.total_number/2:
        team_now = 1
        image = conf.player_image_red
    other_players.add(player.Player(team_now, int(screen_rect.centerx * conf.init_pos[i][0]),
                                    int(screen_rect.centery * conf.init_pos[i][1]), i, image))


ball = Ball(screen_rect.centerx, screen_rect.centery)


# While loop for main logic of the game
while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pressed_keys = pygame.key.get_pressed()
    p1.inputHandler(pressed_keys, ball)

    s.send((compress(p1.id, p1.rect.centerx, p1.rect.centery)).encode('utf8'))

    # this part about catch ball may needed to be dealt in server
    '''
    catched_player = pygame.sprite.spritecollideany(ball, players)
    if catched_player and not(ball.if_catched):
        if catched_player.check_shoot_cd():
            catched_player.catch_ball(ball)
            ball.catched(catched_player)
    '''

    # need to receive message from server

    screen.blit(background, (0, 0))
    for other_player in other_players.sprites():
        other_player.render(screen)
    p1.render(screen)
    ball.render(screen)

    pygame.display.update()

# TODO: Boundary checking for ball
# TODO: Steal ball checking
# TODO: Add a input handler for 2 players game
