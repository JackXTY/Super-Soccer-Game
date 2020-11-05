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

        self.max_time = 60000
        self.total_number = 2
        self.init_pos = [[1.0, 1.0], [0.5, 1.0], [1.5, 1.0]]

        self.ball_cd_time = 1000
        #self.bullet_v = 5
        #self.bullet_cd_time = 3000


def compress(status, id, x, y, time):
    result = "{" + status + '/' + str(id) + '/' + str(x) + '/' + str(y) + '/' + str(time) + '}'
    return result

def compress_ball(id, x, y):
    result = "{" + "Ball" + '/' + str(id) + '/' + str(x) + '/' + str(y) + '}'
    return result

def decompress(target):
    if target[0] == '{' and target[-1] == '}':
        results = target[1:-1].split("}{")
        for i in range(len(results)):
            result = results[i].split('/')
            result[1] = int(result[1])  # id
            result[2] = float(result[2])  # x
            result[3] = float(result[3])  # y
            if result[0] == "Shoot":
                result[4] = int(result[4])  # direction
            if result[0] == "True":
                result[4] = float(result[4])  # time
            results[i] = result
        return results
