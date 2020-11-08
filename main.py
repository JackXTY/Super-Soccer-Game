import pygame
from pygame.locals import *
from pygame.sprite import Group
import sys
import player
from ball import Ball
from config import Config, compress, decompress, compress_update_ball, compress_shoot
import socket

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
recv_datas = []

# While loop for main logic of the game
while game_on:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pressed_keys = pygame.key.get_pressed()
    #print(pressed_keys[K_SPACE])
    p1.input_handler(pressed_keys)

    if p1.shoot_dir < 99:
        #print("press space, dir:" + str(p1.shoot_dir))
        if ball.catcher == p1.id and p1.check_shoot_cd():
            print("try to shoot")
            s.send((compress_shoot(p1.id, p1.rect.centerx, p1.rect.centery, p1.shoot_dir)).encode('utf8'))
            '''
            while True:
                recv_datas = decompress(s.recv(2048).decode('utf8'))
                for recv_data in recv_datas:
                    print(recv_data)
                    if recv_data[0] == "Ball" and recv_data[1] < 0:
                        p1.shoot_update()
                        ball.catcher = -1
                        break
                    else:
                        print("waiting for ball msg")
            '''
            p1.shoot_update()
            ball.catcher = -1
        p1.shoot_dir = 99

    # if collides, catch the ball
    if pygame.sprite.collide_rect(ball, p1) and ball.catcher == -1:
        if p1.check_shoot_cd():  # check if the player just shoot the ball
            s.send((compress_update_ball(p1.id, p1.rect.centerx, p1.rect.centery)).encode('utf8'))
            print(str(p1.id) + " get ball")
            # there could be bug due to packet splicing, but I'm a bit lazy to, hope no bug
            temp_recv_data = decompress(s.recv(2048).decode('utf8'))[0]
            if temp_recv_data[0] == "Ball" and temp_recv_data[1] == p1.id:
                print(temp_recv_data)
                ball.catcher = p1.id
    # if catching the ball, update ball position to server
    elif p1.id == ball.catcher:
        #print(str(p1.id) + " have ball")
        s.send((compress_update_ball(p1.id, p1.rect.centerx, p1.rect.centery)).encode('utf8'))
        temp_recv_data = decompress(s.recv(2048).decode('utf8'))[0]
        if temp_recv_data[0] == "Ball" and temp_recv_data[1] != p1.id:
            ball.catcher = recv_data[1]
            print("ball is stealed by "+str(ball.catcher))

    # update p1 position to server, and receive other players' & ball's positions
    s.send((compress("True", p1.id, p1.rect.centerx, p1.rect.centery, 0.0)).encode('utf8'))

    if_begin = False
    if_end = False
    i = 0
    while not(if_begin):
        recv_datas.extend(decompress(s.recv(2048).decode('utf8')))
        recv_data = recv_datas[0]
        #print(recv_data)
        if recv_data[0] == "Begin_line":
            recv_datas.pop(0)
            if_begin = True
    while not(if_end):
        recv_datas.extend(decompress(s.recv(2048).decode('utf8')))
        while len(recv_datas) > 0:
            recv_data = recv_datas[0]
            #print(recv_data)
            recv_datas.pop(0)
            if recv_data[0] == "End":
                game_on = False
                break
            elif recv_data[0] == "End_line":
                if_end = True
            elif recv_data[0] == "Ball":
                i = i + 1
                if recv_data[1] != ball.catcher:
                    print("catcher changes")
                ball.catcher = recv_data[1]
                ball.rect.centerx = recv_data[2]
                ball.rect.centery = recv_data[3]
            else:  # default case: recv_data[0] == "True"
                i = i + 1
                pid = recv_data[1]
                for player in players:
                    if player.id == pid and pid != p1.id:
                        player.rect.centerx = recv_data[2]
                        player.rect.centery = recv_data[3]
                        break
    if i != N + 1:
        print("CONNECTION ERROR")

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
