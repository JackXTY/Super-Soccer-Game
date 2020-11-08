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
        self.player_power = 50.0
        self.friction = 0.1
        self.shoot_cd_time = 1000

        self.max_time = 60000
        self.total_number = 2
        self.init_pos = [[1.0, 1.0], [0.5, 1.0], [1.5, 1.0]]

        self.ball_cd_time = 1000
        #self.bullet_v = 5
        #self.bullet_cd_time = 3000


class Velocity:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def compress(status, id, x, y, time):
    result = "{" + status + '/' + str(id) + '/' + str(x) + '/' + str(y) + '/' + str(time) + '}'
    return result


def compress_ball(id, x, y):
    result = "{" + "Ball" + '/' + str(id) + '/' + str(x) + '/' + str(y) + '}'
    return result


def compress_update_ball(id, x, y):
    result = "{" + "Ball_update" + '/' + str(id) + '/' + str(x) + '/' + str(y) + '}'
    return result


def compress_shoot(id, x, y, dir):
    result = "{" + "Shoot" + '/' + str(id) + '/' + str(x) + '/' + str(y) + '/' + str(dir) + '}'
    return result


def decompress(target):
    results = []
    if target[0] == '{' and target[-1] == '}':
        results += target[1:-1].split("}{")
        for i in range(len(results)):
            result = results[i].split('/')
            result[1] = int(result[1])  # id
            if not((result[0] == "End_line" or result[0] == "Begin_line") and result[1] == 99):
                result[2] = float(result[2])  # x
                result[3] = float(result[3])  # y
                if result[0] == "Shoot":
                    result[4] = int(result[4])  # direction
                if result[0] == "True":
                    result[4] = float(result[4])  # time
            results[i] = result
        return results


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


def dir_to_xy(dir):
    x = dir % 10
    if x == 2:
        x = -1
    y = dir / 10
    if y == 2:
        y = -1
    return Velocity(x, y)


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