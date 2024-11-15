from environment import environment
from anotherAgent import PPo
import numpy as np
import tensorflow as tf

env = environment(dataset= "optim_train_set.csv")
agent = PPo(env= env, nb_episodes= 250, alpha= 0.01, gamma= 0.5, epsilon= 1, eps_dec= 1e-5, eps_min= 0.1)
"""
current_state = env.reset()
print(current_state)

actions = agent.choose_action(current_state[0])
print(actions)

new_state, rewards, done, allocation_ratio = env.step(actions)
print(new_state)
print(rewards)
print(done)
print(allocation_ratio)
"""
agent.train_agent()