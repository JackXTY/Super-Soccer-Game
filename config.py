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

        self.max_time = 40000
        self.available_player_numbers = [2, 4, 6, 10]
        self.init_pos = {2: [[0.5, 1.0], [1.5, 1.0]], 
                        4: [[0.75, 0.5], [0.75, 1.5], [1.25, 0.5], [1.25, 1.5]],
                        6: [[0.75, 0.5], [0.5, 1.0], [0.75, 1.5], [1.25, 0.5], [1.5, 1.0], [1.25, 1.5]],
                        10: [[0.75, 0.5], [0.625, 0.75], [0.5, 1.0], [0.625, 1.25], [0.75, 1.5], [1.25, 0.5], [1.375, 0.75], [1.5, 1.0], [1.375, 1.25], [1.25, 1.5]]}

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

def rewards_func(r, p_x, p_y, n_x, n_y, N):
    rewards = r
    prev_pos = {}
    prev_pos["x"] = p_x
    prev_pos["y"] = p_y
    new_pos = {}
    new_pos["x"] = n_x
    new_pos["y"] = n_y

    for i in range(int(N/2)):
        for axis in ["x", "y"]:
            if abs(prev_pos[axis][N+int(N/2)] - prev_pos[axis][i]) > abs(new_pos[axis][N] - new_pos[axis][i]):
                rewards[0] += 200
            elif abs(prev_pos[axis][N+int(N/2)] - prev_pos[axis][i]) < abs(new_pos[axis][N] - new_pos[axis][i]):
                rewards[0] -= 200
            else:
                rewards[0] -= 50


    for i in range(int(N/2), N):
        for axis in ["x", "y"]:
            if abs(prev_pos[axis][N-int(N/2)] - prev_pos[axis][i]) > abs(new_pos[axis][N] - new_pos[axis][i]):
                rewards[1] += 200
            elif abs(prev_pos[axis][N-int(N/2)] - prev_pos[axis][i]) < abs(new_pos[axis][N] - new_pos[axis][i]):
                rewards[1] -= 200
            else:
                rewards[1] -= 50

    if new_pos["x"][N] > prev_pos["x"][N]:
        rewards[0] += 600
        rewards[1] -= 600
    elif new_pos["x"][N] < prev_pos["x"][N]:
        rewards[0] -= 600
        rewards[1] += 600  

    #print(rewards)
    return rewards

def new_rewards_func(r, p_x, p_y, n_x, n_y, N):
    rewards = r
    prev_pos = {}
    prev_pos["x"] = p_x
    prev_pos["y"] = p_y
    new_pos = {}
    new_pos["x"] = n_x
    new_pos["y"] = n_y

    sum_0 = 0
    sum_1 = 0
    half = int(N/2)
    dis = []
    for i in range(half):
        dis.append((abs(prev_pos["x"][N] - prev_pos["x"][i])**2 + abs(prev_pos["y"][N] - prev_pos["y"][i])**2)**0.5)
        sum_0 += dis[i]
    for i in range(half, N):
        dis.append((abs(prev_pos["x"][N] - prev_pos["x"][i])**2 + abs(prev_pos["y"][N] - prev_pos["y"][i])**2)**0.5)
        sum_1 += dis[i]

    sum_0 /= half
    sum_1 /= half

    for i in range(half):
        rewards[0] += (600 - dis[i])
    for i in range(half, N):
        rewards[1] += (600 - dis[i])

    # if sum_0 > sum_1:
    #     rewards[0] += 300
    #     rewards[1] -= 300
    # elif sum_1 > sum_0:
    #     rewards[0] -= 300
    #     rewards[1] += 300  

    #print(rewards)
    return rewards
