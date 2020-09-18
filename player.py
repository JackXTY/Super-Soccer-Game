import pygame
from config import Config

conf = Config()

class Velocity:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0

class Player_state:
    def __init__(self, initial_pos_x, initial_pos_y, player_image):
        self.v = Velocity()
        self.player = pygame.image.load(player_image)
        self.rect = self.player.get_rect()
        self.rect.centerx = initial_pos_x
        self.rect.centery = initial_pos_y
        self.timer = pygame.time.Clock()
        self.timer.tick()
        self.cd_time = conf.bullet_cd_time

    def update_pos(self):
        left_bound = conf.width * 0.13
        right_bound = conf.width * 0.87
        upper_bound = conf.height * 0.13
        lower_bound = conf.height * 0.87
        pos_x = self.rect.centerx + self.v.x
        pos_y = self.rect.centery + self.v.y
        if(pos_x < left_bound):
            pos_x = left_bound
        if (pos_x > right_bound):
            pos_x = right_bound
        if (pos_y < upper_bound):
            pos_y = upper_bound
        if (pos_y > lower_bound):
            pos_y = lower_bound
        self.rect.centerx = int(pos_x)
        self.rect.centery = int(pos_y)

    def render(self, screen):
        self.update_pos()
        screen.blit(self.player, self.rect)

    def shoot(self):
        self.timer.tick()
        if(self.timer.get_time() >= self.cd_time):
            self.cd_time = conf.bullet_cd_time
            return True
        else:
            self.cd_time = self.cd_time - self.timer.get_time()
            return False
