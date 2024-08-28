import gymnasium as gym
from src.train import train
from src.test import test
from src.training_settings import Settings

def main():
    env = gym.make("LunarLander-v2")
    settings = Settings(episodes=1000, epsilon=1, ep_decay=.99, gamma=.99, lr=.001, buffer_size=1000, batch_size=64, test_render=True)
    train(env, settings)
    test(env, settings)
    pass

if __name__ == "__main__":
    main()