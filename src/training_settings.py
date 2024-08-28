
class Settings:
    def __init__(self, episodes, epsilon, ep_decay, gamma, lr, buffer_size, batch_size, test_render):
        self.episodes = episodes
        self.epsilon = epsilon
        self.ep_decay = ep_decay
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.test_render = test_render

    def eps_decay(self):
        self.epsilon *= self.ep_decay