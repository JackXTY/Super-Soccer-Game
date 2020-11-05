import pygame
from pygame.locals import *
from pygame.sprite import Group
import sys
import player
from ball import Ball
from config import Config, compress, decompress, compress_ball
import socket
import select

pygame.init()

conf = Config()
N = conf.total_number

screen = pygame.display.set_mode(conf.size)
screen_rect = screen.get_rect()
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load(conf.background_image).convert()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 6666))


pre_data = decompress(s.recv(2048).decode('utf8'))[0]
print(pre_data)
team_now = 0
image = conf.player_image_blue
if pre_data[1] > N/2:
    team_now = 1
    image = conf.player_image_red
print("my_player: id={}, team={}".format(pre_data[1], team_now))
p1 = player.Player(0, int(screen_rect.centerx*pre_data[2]), int(screen_rect.centery*pre_data[3]),
                   pre_data[1], image)

# read other player data directly from config
players = Group()
for i in range(1, N+1):
    if i == p1.id:
        players.add(p1)
        continue
    team_now = 0
    image = conf.player_image_blue
    if i > N/2:
        team_now = 1
        image = conf.player_image_red
    print("other_player: id={}, team={}".format(i, team_now))
    players.add(player.Player(team_now, int(screen_rect.centerx * conf.init_pos[i][0]),
                                    int(screen_rect.centery * conf.init_pos[i][1]), i, image))


ball = Ball(screen_rect.centerx, screen_rect.centery)


# While loop waiting for all client to be ready
while True:
    s.send((compress("False", p1.id, p1.rect.centerx, p1.rect.centery, 0.0)).encode('utf8'))
    recv_data = decompress(s.recv(2048).decode('utf8'))[0]
    if recv_data[0] == "Begin":
        break


game_on = True

# While loop for main logic of the game
while game_on:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pressed_keys = pygame.key.get_pressed()
    p1.inputHandler(pressed_keys, ball)

    # catch the ball
    if pygame.sprite.collide_rect(ball, p1):
        if p1.check_shoot_cd():  # check if the player just shoot the ball
            s.send((compress_ball(p1.id, p1.rect.centerx, p1.rect.centery)).encode('utf8'))
            print(str(p1.id) + " get ball")
            # there could be bug due to packet splicing, but I'm a bit lazy to, hope no bug
            recv_data = decompress(s.recv(2048).decode('utf8'))[0]
            if recv_data[0] == "Ball" and recv_data[1] == p1.id:
                p1.catch_ball(ball)
    elif p1.id == ball.catcher:  # update ball position to server
        print(str(p1.id) + " have ball")
        s.send((compress_ball(p1.id, p1.rect.centerx, p1.rect.centery)).encode('utf8'))
        recv_data = decompress(s.recv(2048).decode('utf8'))[0]
        if recv_data[0] == "Ball" and recv_data[1] != p1.id:
            ball.catcher = recv_data[1]

    s.send((compress("True", p1.id, p1.rect.centerx, p1.rect.centery, 0.0)).encode('utf8'))
    i = 0
    while i < conf.total_number:
        recv_datas = decompress(s.recv(2048).decode('utf8'))
        #print(recv_datas)
        for recv_data in recv_datas:
            i = i + 1
            if recv_data[0]=="End":
                game_on = False
                break
            elif recv_data[0] == "Ball":
                ball.catcher = recv_data[1]
                ball.rect.centerx = recv_data[2]
                ball.rect.centery = recv_data[3]
            else:
                pid = recv_data[1]
                for player in players:
                    if player.id == pid and pid != p1.id:
                        player.rect.centerx = recv_data[2]
                        player.rect.centery = recv_data[3]
                        break

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


# should be implemented in server.py
# TODO: Boundary checking for ball
# TODO: Steal ball checking
