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
        self.player_power = 5.0
        self.friction = 0.1
        self.shoot_cd_time = 300

        self.total_number = 2
        self.init_pos = [[1.0, 1.0], [0.5, 1.0], [1.5, 1.0]]
        #self.bullet_v = 5
        #self.bullet_cd_time = 3000

def compress(id, x, y):
    result = '[' + str(id) + '/' + str(x) + '/' + str(y) + ']'
    return result

def decompress(target):
    if target[0]=='[' and target[-1]==']':
        result = target[1:-1].split('/')
        result[0] = int(result[0])
        result[1] = float(result[1])
        result[2] = float(result[2])
    return result