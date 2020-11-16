import pygame
from pygame.locals import *
from pygame.sprite import Sprite
from config import Config, xy_to_dir, Velocity

conf = Config()


class Player(Sprite):
    def __init__(self, team, initial_pos_x, initial_pos_y, pid, player_image):
        super(Player, self).__init__()
        self.id = pid
        self.team = team  # team-0: attack right door / team-1: attack left door
        self.v = Velocity(0.0, 0.0)
        self.player_image = pygame.image.load(player_image)
        self.rect = self.player_image.get_rect()
        self.rect.centerx = initial_pos_x
        self.rect.centery = initial_pos_y
        self.timer = pygame.time.Clock()
        self.cd_time = conf.shoot_cd_time
        self.shoot_dir = 99

    def update(self):
        pos_x = self.rect.centerx + self.v.x
        pos_y = self.rect.centery + self.v.y

        left_bound = conf.width * 0.125
        right_bound = conf.width * 0.875
        upper_bound = conf.height * 0.125
        lower_bound = conf.height * 0.875
        if pos_x < left_bound:
            pos_x = left_bound
        if pos_x > right_bound:
            pos_x = right_bound
        if pos_y < upper_bound:
            pos_y = upper_bound
        if pos_y > lower_bound:
            pos_y = lower_bound

        self.rect.centerx = int(pos_x)
        self.rect.centery = int(pos_y)

    def render(self, screen):
        self.update()
        screen.blit(self.player_image, self.rect)

    def shoot_update(self):
        self.timer.tick()
        self.cd_time = conf.shoot_cd_time

    def check_shoot_cd(self):
        if self.timer.tick() > self.cd_time:
            return True
        else:
            self.cd_time = self.cd_time - self.timer.get_time()
            return False

    def input_handler(self, pressed_keys):
        if pressed_keys[K_LEFT]:
            self.v.x = -conf.player_v
        elif pressed_keys[K_RIGHT]:
            self.v.x = conf.player_v
        else:
            self.v.x = 0
        if pressed_keys[K_UP]:
            self.v.y = -conf.player_v
        elif pressed_keys[K_DOWN]:
            self.v.y = conf.player_v
        else:
            self.v.y = 0
        if pressed_keys[K_SPACE]:
            self.shoot_dir = xy_to_dir(self.team, self.v.x, self.v.y)
        else:
            self.shoot_dir = 99
