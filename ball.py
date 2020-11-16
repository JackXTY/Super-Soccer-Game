import pygame
from pygame.sprite import Sprite
from config import Config, dir_to_xy, update_v, Velocity

conf = Config()

class Ball(Sprite):
    def __init__(self, initial_pos_x, initial_pos_y):
        super(Ball, self).__init__()
        self.ball = pygame.image.load(conf.ball_image)
        self.rect = self.ball.get_rect()
        self.rect.centerx = initial_pos_x
        self.rect.centery = initial_pos_y
        self.v = Velocity(0.0, 0.0)  # v = v - at
        self.if_catched = False
        self.catcher = -1
        self.timer = pygame.time.Clock()
        self.remain_time = 0.0

    def shoot_ball(self, x, y, dir):
        self.if_catched = False
        self.catcher = -1
        self.rect.centerx = x
        self.rect.centery = y
        self.v = dir_to_xy(dir)
        if self.v.x == 0:
            self.v.y = self.v.y * conf.player_power
        elif self.v.y == 0:
            self.v.x = self.v.x * conf.player_power
        else:
            self.v.x = self.v.x * (conf.player_power ** 0.5)
            self.v.y = self.v.y * (conf.player_power ** 0.5)

    def update_pos(self):
        if self.v.x != 0 or self.v.y != 0:
            self.rect.centerx = self.rect.centerx + int(self.v.x)
            self.rect.centery = self.rect.centery + int(self.v.y)
            self.v.x = update_v(self.v.x, conf.friction)
            self.v.y = update_v(self.v.y, conf.friction)

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

    def in_door(self):
        if 3.5/15 * conf.height < self.rect.centery < 11.5/15 * conf.height:
            if self.rect.centerx < conf.width * 0.125:
                return 1
            if self.rect.centerx > conf.width * 0.875:
                return 0
        return -1
