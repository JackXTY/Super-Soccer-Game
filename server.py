import socket
import traceback
import datetime
import pygame
from threading import Thread
from config import Config, compress, decompress

class Server:

    __user_class = None

    @staticmethod
    def print_log(message):
        print("[" + str(datetime.datetime.now()) + "]" + message)

    def __init__(self, ip, port):
        self.connection_pool = []  # all connections here
        self.print_log('waiting for server to start..')
        try:
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Listener to accept new socket
            self.listener.bind((ip, port))
            self.listener.listen(5)  # Max waiting number
        except:
            self.print_log('Fail to start server, please check if ip is occupied. Reason：\n' + traceback.format_exc())

        if self.__user_class is None:
            self.print_log('Fail to start server, user self-defined class is not registered')
            return

        self.print_log('Succeed in start server: {}: {}'.format(ip, port))
        while True:
            client, _ = self.listener.accept()  # wait for client side
            user = self.__user_class(client, self.connection_pool)
            self.connection_pool.append(user)
            self.print_log('New connection comes，no. of connections now：{}'.format(len(self.connection_pool)))

    @classmethod
    def register_class(cls, player_class):
        # register self-defined class for user
        cls.__user_class = player_class

conf = Config()


class Data_controller:
    def __init__(self):
        self.max_id = conf.total_number
        self.id_now = 0
        self.pos = conf.init_pos  # pos[0] reserve for ball, other for playerss(i)
        self.ball_catcher = 0  # represent which player catch the ball
        self.timer = pygame.time.Clock()
        self.remain_time = conf.max_time

    def distribute_player_id(self):
        if self.id_now >= self.max_id:
            raise ConnectionError
        self.id_now = self.id_now + 1
        return self.id_now

    def check_time(self):
        time_interval = self.timer.tick()
        self.remain_time -= time_interval
        return self.remain_time

data_cont = Data_controller()


# server won't send msg to the client actively, except assigning id to client
@Server.register_class
class Player_connection():

    def __init__(self, client, connection_pool):
        self.socket_client = client
        self.connection_pool = connection_pool
        self.data_handler()
        self.id = data_cont.distribute_player_id()
        self.x = data_cont.pos[self.id][0]
        self.y = data_cont.pos[self.id][1]
        self.send_data("False", self.id, self.x, self.y, 0.0)  # assign id to client

    def data_handler(self):
        # build an independent thread for every connection
        thread = Thread(target=self.receive_data)
        thread.setDaemon(True)
        thread.start()

    def receive_data(self):
        # receive data
        try:
            while True:
                data = self.socket_client.recv(2048)  # size of each data bag should be less than 2048KB
                if len(data) == 0:
                    self.socket_client.close()
                    self.connection_pool.remove(self) # remove connection
                    break
                # deal with data
                self.deal_recv_data(data)
        except:
            self.connection_pool.remove(self)
            Server.print_log('Bug in receiving data for customer side：\n' + traceback.format_exc())

    def send_data(self, status, player_id, x, y, time):
        message = compress(status, player_id, x, y, time)
        print(message)
        self.socket_client.send(message.encode('utf8'))

    def deal_recv_data(self, data):
        client_data = decompress(data.decode('utf8'))

        if(client_data[0] == "True"):  # game on
            game_time = data_cont.check_time()
            if game_time <= 0:  # game end
                self.send_data("End", self.id, self.x, self.y, 0.0)
            else:
                self.x = client_data[2]
                self.y = client_data[3]
                # send data of this client to other clients
                for connection in self.connection_pool:
                    self.send_data("True", connection.id, connection.x, connection.y, game_time)

        else:  # game not start
            if (len(self.connection_pool) == conf.total_number):
                self.send_data("Begin", self.id, self.x, self.y, 0.0)
                data_cont.timer.tick()
            else:
                self.send_data("False", self.id, self.x, self.y, 0.0)

Server('127.0.0.1', 6666)

# TODO: modify in form of code