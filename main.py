import pygame
from pygame.locals import *
from pygame.sprite import Sprite, Group
import sys
import player
from config import Config

pygame.init()

conf = Config()

screen = pygame.display.set_mode(conf.size)
screen_rect = screen.get_rect()
pygame.display.set_caption('Super Soccer Game')
background = pygame.image.load(conf.background_image).convert()

p = player.Player_state(int(screen_rect.centerx/2), screen_rect.centery, conf.player_image_blue)


# Ball
class Ball_state():
    def __init__(self, initial_pos_x, initial_pos_y):
        self.ball = pygame.image.load(conf.ball_image)
        self.rect = self.ball.get_rect()
        self.rect.centerx = initial_pos_x
        self.rect.centery = initial_pos_y

    def update_pos(self):
        self.rect.centerx = self.rect.centerx
        # Do sth here

    def render(self):
        self.update_pos()
        screen.blit(self.ball, self.rect)

b = Ball_state(screen_rect.centerx, screen_rect.centery)


# Bullet
bullets = Group()

class Bullet(Sprite):
    def __init__(self, pos_x, pos_y):
        super(Bullet, self).__init__()
        self.rect = pygame.Rect(0, 0, 5, 5)
        self.rect.centerx = pos_x
        self.rect.centery = pos_y
        self.bullet = pygame.draw.rect(screen, (255, 0, 0), self.rect)

    def update_pos(self):
        self.rect.centerx = self.rect.centerx

    def render(self):
        self.update_pos()

def create_bullet(if_shoot, pos_x, pos_y):
    if (if_shoot):
        new_bullet = Bullet(pos_x, pos_y)
        bullets.add(new_bullet)
        print('create bullet');

def update_bullet():
    for b in bullets.copy():
        if ((b.rect.centerx < 0) or (b.rect.centery < 0) or
        (b.rect.centerx > conf.width) or (b.rect.centery > conf.height)):
            bullets.remove(b)


# While loop for main logic of the game
while 1:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

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
                create_bullet(p.shoot(), p.rect.centerx, p.rect.centery)

        if event.type == KEYUP:
            if event.key == K_LEFT or event.key == K_RIGHT:
                p.v.x = 0
            if event.key == K_UP or event.key == K_DOWN:
                p.v.y = 0

    screen.blit(background, (0, 0))
    p.render(screen)
    b.render()
    for b in bullets:
        b.render()
    pygame.display.update()
