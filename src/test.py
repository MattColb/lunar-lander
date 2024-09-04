from src.model_interaction import ModelInteraction

#Implement latest
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

