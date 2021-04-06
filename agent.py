class Agent():
    def __init__(self, id, game_mode, model_root_path="./model/", train=True):
        self.id = id
        self.type = "Agent"
        self.game_mode = game_mode
        self.model_path = model_root_path + self.type + \
            "/" + game_mode + "/" + str(id) + ".model"
        self.model = self.load_model()

    def set_state(self, state):
        pass

    def get_state(self, state):
        pass

    def update(self):
        pass

    def make_decision(self, random):
        pass

    def reset(self):
        pass

    def load_model(self):
        pass

    def update_greedy(self):
        pass

    def save_model(self):
        pass