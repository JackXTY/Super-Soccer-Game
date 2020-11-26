class Config:
    def __init__(self):
        self.width = 886
        self.height = 620
        self.size = (self.width, self.height)
        self.player_image_red = 'image/player1.png'
        self.player_image_blue = 'image/player2.png'
        self.ball_image = 'image/ball.png'
        self.background_image = 'image/background.png'

        self.player_v = 5
        self.player_power = 7
        self.friction = 0.14
        self.shoot_cd_time = 1000

        self.max_time = 400000
        self.total_number = 2
        self.init_pos = [[1.0, 1.0], [0.5, 1.0], [1.5, 1.0]]

        self.ball_cd_time = 1000
        #self.bullet_v = 5
        #self.bullet_cd_time = 3000

conf = Config()

class Velocity:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def xy_to_dir(team, x, y):
    # 01-Right, 11-RightUp, 10-Up, 12-LeftUp, 02-Left, 22-LeftDown, 20-Down, 21-RightDown
    res = 0
    if x > 0:
        res = res + 1
    elif x < 0:
        res = res + 2
    if y > 0:
        res = res + 10
    elif y < 0:
        res = res + 20
    if res != 0:
        return res
    # default dir when no other keyboard input
    if team == 1:
        return 2
    else:
        return 1


def dir_to_xy(d):
    # 01-Right, 11-RightUp, 10-Up, 12-LeftUp, 02-Left, 22-LeftDown, 20-Down, 21-RightDown
    x = 0
    y = 0
    if d == 1:
        x = conf.player_power
    elif d == 2:
        x = -conf.player_power
    elif d == 10:
        y = -conf.player_power
    elif d == 20:
        y = conf.player_power
    elif d == 11:
        x = conf.player_power * 0.78
        y = - conf.player_power * 0.78
    elif d == 12:
        x = -conf.player_power * 0.78
        y = - conf.player_power * 0.78
    elif d == 21:
        x = conf.player_power * 0.78
        y = conf.player_power * 0.78
    elif d == 22:
        x = - conf.player_power * 0.78
        y = conf.player_power * 0.78
    return x, y


def update_v(v, f):
    if int(v) == 0:
        return 0
    if v > 0:
        v = v - f
        if v <= 0:
            v = 0
    elif v < 0:
        v = v + f
        if v >= 0:
            v = 0
    return v