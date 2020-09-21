class Config:
    def __init__(self):
        self.width = 886
        self.height = 620
        self.size = (self.width, self.height)
        self.player_image_red = 'image/player1.png'
        self.player_image_blue = 'image/player2.png'
        self.ball_image = 'image/ball.png'
        self.background_image = 'image/background.png'

        self.player_v = 1
        self.player_power = 12.0
        self.friction = 0.1
        self.shoot_cd_time = 300
        #self.bullet_v = 5
        #self.bullet_cd_time = 3000