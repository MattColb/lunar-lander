from src.training_settings import Settings
from src.model_interaction import ModelInteraction
import time

#Implement loading the previous model

def train(env, settings:Settings):
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]

    latest_timestamp = time.time()
    interaction = ModelInteraction(env, settings, observation_space, action_space)
    if settings.latest_model != None:
        interaction.load_model(f"./saved_models/{settings.latest_model}")

    total_episodes = settings.episodes

    while settings.episodes != 0: 
        state, info = env.reset()
        current_reward = 0
        truncated, terminated = False, False
        while not (truncated or terminated):
            action = interaction.act(state)
            new_state, reward, truncated, terminated, info = env.step(action)
            interaction.memory.add((state, action, reward, truncated or terminated, new_state))
            current_reward += reward
            state = new_state
            
            interaction.train()


        print(f"Episode {settings.original_episodes - settings.episodes}: {current_reward} reward")
        #Update target after a certain period
        #Also save current settings?
        #Find a way to check for the latest model + settings to load in the next time that you want to run?
        #Saves model every 10 minutes
        if time.time() - latest_timestamp >= 600:
            interaction.update_target_model()
            time_format = time.strftime("%Y-%m-%d_%H-%M-%S")
            interaction.save_model(f"./saved_models/{time_format}.h5")
            settings.save_settings(f"./saved_settings/{time_format}.json")
            latest_timestamp = time.time()
        
        settings.episodes -= 1
        settings.eps_decay()

    interaction.save_model(f"./saved_models/final_model.h5") 
