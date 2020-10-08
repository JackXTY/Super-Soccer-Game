import socket
import traceback
import datetime
from threading import Thread

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
            self.print_log('Fail to start server, user seld-defined class is not registered')
            return

        self.print_log('Succeed in start server{}:{}'.format(ip, port))
        while True:
            client, _ = self.listener.accept()  # wait for client side
            user = self.__user_class(client, self.connection_pool)
            self.connection_pool.append(user)
            self.print_log('New connection comes，no. of connections now：{}'.format(len(self.connection_pool)))

    @classmethod
    def register_cls(cls, sub_cls):
        # register self-defined class for user
        if not issubclass(sub_cls, Connection):
            cls.print_log('Fail to register self-defined user class, type not matched')
            return
        cls.__user_class = sub_cls


class Connection:

    def __init__(self, socket, connections):
        self.socket = socket
        self.connections = connections
        self.data_handler()

    def data_handler(self):
        # build an independent thread for every connection
        thread = Thread(target=self.recv_data)
        thread.setDaemon(True)
        thread.start()

    def recv_data(self):
        # receive data
        try:
            while True:
                bytes = self.socket.recv(2048)  # size of each data bag should be less than 2038KB
                if len(bytes) == 0:
                    self.socket.close()
                    # remove connection
                    self.connections.remove(self)
                    break
                # deal with data
                self.deal_data(bytes)
        except:
            self.connections.remove(self)
            Server.print_log('Bug in receiving data for customer side：\n' + traceback.format_exc())

    def deal_data(self, bytes):
        # to be overridden by child classes
        raise NotImplementedError


@Server.register_cls
class Player(Connection):

    def __init__(self, *args):
        super().__init__(*args)
        self.x = None
        self.y = None

    def deal_data(self, bytes):
        print('\nMessage:', bytes.decode('utf8'))

Server('127.0.0.1', 6666)

# TODO: modify in form of code