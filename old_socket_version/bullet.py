import pygame
from pygame.sprite import Sprite
from config import Config

conf = Config()

class Bullet(Sprite):
    def __init__(self, pos_x, pos_y, screen):
        super(Bullet, self).__init__()
        self.rect = pygame.Rect(0, 0, 5, 5)
        self.rect.centerx = pos_x
        self.rect.centery = pos_y
        self.bullet = pygame.draw.rect(screen, (255, 0, 0), self.rect)

    def update_pos(self):
        self.rect.centerx = self.rect.centerx

    def render(self):
        self.update_pos()
'''
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
'''

'''
    # shoot for player
    def shoot(self):
        if self.if_catch:
            self.timer.tick()
            if(self.timer.get_time() >= self.cd_time):
                self.cd_time = conf.bullet_cd_time
                return True
            else:
                self.cd_time = self.cd_time - self.timer.get_time()
                return False
'''