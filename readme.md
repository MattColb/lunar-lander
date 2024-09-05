# OpenAI Gymnasium Lunar Lander Environment

This repository contains the code that I used to train a model to complete the lunar lander openAI gymnasium environment.

This program runs by creating a Settings object, which contains a set of parameters that are used to train the model. These parameters are:
- episodes (int) - the total number of times that you want the environment to train on 
- epsilon (float) - the starting chance that the model will choose a random action vs the best action (recommended to be 1.00)
- ep_decay (float) - The amount that you want epsilon to decay across each episode
- gamma (float) - the modifier for the reward updates that are done on experience replay
- lr (float) - the learning rate for the model.  
- buffer_size (int) - the total number of experiences to be saved in experience replay 
- batch_size (int) - the number of experiences to be used each time you do an experience replay
- test_render (bool) - Says whether or not you want to render the environment in a visible way when you are testing 
- only_test (bool) - Only runs the test portion on the most recent checkpoint
- use_previous (bool) - Says whether or not you want to use previous checkpoints or completely clear out any history

The final_model.h5 in this repository contains the model after training using the settings that are included in the main.py.

When testing the model over 10 episodes, it scored an average of __. 
