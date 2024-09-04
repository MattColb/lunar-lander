import os
import time
import json

class Settings:
    def __init__(self, episodes, epsilon, ep_decay, gamma, lr, buffer_size, 
                 batch_size, test_render=True, only_test=False, use_previous=True, 
                 checkpoint_seconds=600):
        self.episodes = episodes
        self.original_episodes = episodes
        self.epsilon = epsilon
        self.ep_decay = ep_decay
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.test_render = test_render
        self.only_test = only_test
        self.checkpoint_seconds = checkpoint_seconds
        self.latest_settings = None
        self.latest_model = None
        if use_previous == False:
            models = os.listdir("./saved_models")
            settings = os.listdir("./saved_settings")
            for model in models: os.remove(f"./saved_models/{model}") 
            for setting in settings: os.remove(f"./saved_settings/{setting}")
        if use_previous:
            self.get_latest_model()
            self.get_latest_settings()
            if self.latest_settings != None:
                self.load_settings(self.latest_settings)

    def eps_decay(self):
        self.epsilon *= self.ep_decay

    #If final_model is in, your model list, just use that
    def get_latest_model(self):
        files = os.listdir("./saved_models")
        sorted_files = sorted(files)[::-1]
        #Check that this works
        if "final_model.h5" in sorted_files:
            self.latest_model = "final_model.h5"
        elif len(sorted_files) != 0:
            self.latest_model = sorted_files[0]
        
    def save_settings(self, file_path):
        all_settings = {"episodes":self.episodes, 
                        "original_episodes":self.original_episodes,
                        "epsilon":self.epsilon, 
                        "ep_decay":self.ep_decay,
                        "gamma":self.gamma, 
                        "lr":self.lr, 
                        "buffer_size":self.buffer_size, 
                        "batch_size":self.batch_size,
                        "test_render":self.test_render, 
                        "only_test":self.only_test,
                        "checkpoint_seconds":self.checkpoint_seconds}
        
        with open(file_path, "w") as f:
            json.dump(all_settings, f)

    def load_settings(self, file_path):
        with open(f"./saved_settings/{file_path}", "r") as f:
            all_settings = json.load(f)

        self.episodes = all_settings["episodes"]
        self.original_episodes = all_settings["original_episodes"]
        self.epsilon = all_settings["epsilon"]
        self.ep_decay = all_settings["ep_decay"]
        self.gamma = all_settings["gamma"]
        self.lr = all_settings["lr"]
        self.buffer_size = all_settings["buffer_size"]
        self.batch_size = all_settings["batch_size"]
        self.test_render = all_settings["test_render"]
        self.only_test = all_settings["only_test"]
        self.checkpoint_seconds = all_settings["checkpoint_seconds"]

    def get_latest_settings(self):
        files = os.listdir("./saved_settings")
        sorted_files = sorted(files)[::-1]
        if len(sorted_files) != 0:
            self.latest_settings = sorted_files[0]