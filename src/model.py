import tensorflow as tf
from src.training_settings import Settings
from tensorflow.keras import models, layers, optimizers

#Add something so that the structure can adjust

def make_model(observation_space, action_space, settings:Settings):
    model = models.Sequential()
    model.add(layers.Dense(128, activation="relu", input_dim=observation_space))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(action_space, activation="linear"))
    model.compile(optimizer=optimizers.Adam(learning_rate = settings.lr), loss="mse")
    return model