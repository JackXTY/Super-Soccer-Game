import pygame
from pygame.sprite import Sprite
from config import Config

conf = Config()

class Ball(Sprite):
    def __init__(self, initial_pos_x, initial_pos_y):
        super(Ball, self).__init__()
        self.ball = pygame.image.load(conf.ball_image)
        self.rect = self.ball.get_rect()
        self.rect.centerx = initial_pos_x
        self.rect.centery = initial_pos_y
        self.v = 0.  # v = v - at, only in x-axis
        self.direction = 0
        self.if_catched = False
        self.catcher = -1
        self.timer = pygame.time.Clock()
        self.remain_time = 0.0

    def shoot(self, team, power):
        self.if_catched = False
        print("Team "+str(team)+" shoot")
        self.v = power
        if (team == 0):
            self.direction = 1
        else:
            self.direction = -1

    def update_pos(self):
        self.rect.centerx = self.rect.centerx + int(self.direction * self.v)

        if self.v > 0:
            self.v = self.v - conf.friction
            if self.v <= 0:
                self.v = 0

    def render(self, screen):
        self.update_pos()
        screen.blit(self.ball, self.rect)

    def check_ball(self, new_id, new_x, new_y):
        if self.catcher < 0:
            self.if_catched = True
            self.catcher = new_id
            self.rect.centerx = new_x
            self.rect.centery = new_y
            self.timer.tick()
            self.remain_time = conf.ball_cd_time
        elif self.catcher == new_id:
            self.rect.centerx = new_x
            self.rect.centery = new_y
        else:
            self.remain_time = self.remain_time - self.timer.tick()
            if self.remain_time < 0 and new_id != self.catcher:
                self.catcher = new_id
                self.rect.centerx = new_x
                self.rect.centery = new_y
                self.timer.tick()
                self.remain_time = conf.ball_cd_time
