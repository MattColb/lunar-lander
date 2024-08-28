from src.model import make_model
import tensorflow as tf
import numpy as np
from src.memory import RecallMemory
class ModelInteraction:
    def __init__(self, env, settings, observation_space, action_space):
        self.env = env
        self.settings = settings
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = RecallMemory(settings)
        self.model = make_model(observation_space, action_space, settings)
        self.target_model = make_model(observation_space, action_space, settings)
        self.update_target_model()

    def act(self, state, test=False):
        if np.random.rand() <= self.settings.epsilon and test == False:
            return self.env.action_space.sample()
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self):
        if len(self.memory) < self.settings.batch_size:
            return
        
        samples = self.memory.sample()

        states, actions, rewards, dones, new_states = zip(*samples)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=int)
        new_states = np.array(new_states)

        target_q_values = self.target_model.predict(new_states, verbose=0)
        targets = rewards + self.settings.gamma * np.max(target_q_values, axis=1) * (1-dones)
        current_q_values = self.model.predict(states, verbose=0)

        for i, action in enumerate(actions):
            current_q_values[i][action] = targets[i]

        self.model.fit(states, current_q_values, epochs=1, verbose=0)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)