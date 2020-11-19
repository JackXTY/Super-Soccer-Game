import pygame

class Text():
    def __init__(self):
        self.font = pygame.font.Font("AdobeFanHeitiStd-Bold.otf", 30)

    def render(self, screen, score, game_time):
        info = str(score) + "  time:" + str(game_time)
        text = self.font.render(info, True, (255, 255, 255))
        screen.blit(text, (screen.get_rect().centerx - text.get_rect().centerx, 0))


