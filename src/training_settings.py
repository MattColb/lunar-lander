import os
import time
import json

class Settings:
    def __init__(self, episodes, epsilon, ep_decay, gamma, lr, buffer_size, batch_size, test_render=True, only_test=False):
        self.episodes = episodes
        self.epsilon = epsilon
        self.ep_decay = ep_decay
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.test_render = test_render
        self.only_test = only_test
        self.latest_settings = None
        self.latest_model = None
        self.get_latest_model()
        self.get_latest_settings()
        if self.latest_settings != None:
            self.load_settings(self.latest_settings)

    def eps_decay(self):
        self.epsilon *= self.ep_decay

    def get_latest_model(self):
        files = os.listdir("../saved_models")
        sorted_files = sorted(files)
        if len(sorted_files) != 0:
            self.latest_model = sorted_files[0]
        
    def save_settings(self, file_path):
        all_settings = {"episodes":self.episodes, 
                        "epsilon":self.epsilon, 
                        "ep_decay":self.ep_decay,
                        "gamma":self.gamma, 
                        "lr":self.lr, 
                        "buffer_size":self.buffer_size, 
                        "batch_size":self.batch_size,
                        "test_render":self.test_render, 
                        "only_test":self.only_test}
        
        with open(file_path, "w") as f:
            json.dump(all_settings, f)

    def load_settings(self, file_path):
        with open(file_path, "r") as f:
            all_settings = json.load(f)

        self.episodes = all_settings["episodes"]
        self.epsilon = all_settings["epsilon"]
        self.ep_decay = all_settings["ep_decay"]
        self.gamma = all_settings["gamma"]
        self.lr = all_settings["lr"]
        self.buffer_size = all_settings["buffer_size"]
        self.batch_size = all_settings["batch_size"]
        self.test_render = all_settings["test_render"]
        self.only_test = all_settings["only_test"]

    def get_latest_settings(self):
        files = os.listdir("../saved_settings")
        sorted_files = sorted(files)
        if len(sorted_files) != 0:
            self.latest_settings = sorted_files[0]

if __name__ == "__main__":
    settings = Settings(episodes=1000, epsilon=1, ep_decay=.99, gamma=.99, lr=.001, buffer_size=1000, batch_size=64, test_render=True)
    settings.save_settings("../saved_settings/2024-09-03_08-33-40.json")