import pygame
from pygame.locals import *
import sys

pygame.init()

size = width, height = 886, 620
speed = [2, 2]
black = 0, 0, 0

screen = pygame.display.set_mode(size)
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load('image/background.png')

player_image = 'image/player1.png'
ball_image = 'image/ball.png'

player_velocity = 1

class Velocity:
    def __init__(self):
        self.x = 0
        self.y = 0

class Player_state:
    def __init__(self, initial_pos_x, initial_pos_y):
        self.v = Velocity()
        self.pos_x = initial_pos_x
        self.pos_y = initial_pos_y
        self.player = pygame.image.load(player_image)

    def update_pos(self):
        left_bound = width * 0.1 + 5
        right_bound = width * 0.9 - 58
        upper_bound = height * 0.1 + 14
        lower_bound = height * 0.9 - 30
        self.pos_x = self.pos_x + self.v.x
        self.pos_y = self.pos_y + self.v.y

        if(self.pos_x < left_bound):
            self.pos_x = left_bound
        if (self.pos_x > right_bound):
            self.pos_x = right_bound
        if (self.pos_y < upper_bound):
            self.pos_y = upper_bound
        if (self.pos_y > lower_bound):
            self.pos_y = lower_bound

    def render(self):
        self.update_pos()
        screen.blit(self.player, (int(self.pos_x), int(self.pos_y)))

p = Player_state(width/4-8, height/2-16)


class Ball_state():
    def __init__(self):
        self.ball = pygame.image.load(ball_image)

    def update_pos(self):
        # realize the collision system

    def render(self):
        self.update_pos()


while 1:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                p.v.x = -player_velocity
            elif event.key == K_RIGHT:
                p.v.x = player_velocity
            if event.key == K_UP:
                p.v.y = -player_velocity
            elif event.key == K_DOWN:
                p.v.y = player_velocity

        if event.type == KEYUP:
            if event.key == K_LEFT or event.key == K_RIGHT:
                p.v.x = 0
            if event.key == K_UP or event.key == K_DOWN:
                p.v.y = 0

    screen.blit(background, (0, 0))
    p.render()
    pygame.display.update()
