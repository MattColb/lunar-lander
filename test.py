from src.model_interaction import ModelInteraction
from src.training_settings import Settings
import gymnasium as gym

def test(env, settings):
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]

    interaction = ModelInteraction(env, settings, observation_space, action_space)
    interaction.load_model("./saved_models/final_model.h5")

    for i in range(10):
        total_reward = 0
        truncated, terminated = False, False
        state, info = env.reset()
        while not (truncated or terminated):
            action = interaction.act(state, test=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            state = new_state
            total_reward += reward

        print(f"Episode {i+1}: {total_reward} reward")

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    settings = Settings(episodes=1000, epsilon=1, ep_decay=.99, gamma=.99, lr=.001, buffer_size=1000, batch_size=64, test_render=True)
    test(env, settings)