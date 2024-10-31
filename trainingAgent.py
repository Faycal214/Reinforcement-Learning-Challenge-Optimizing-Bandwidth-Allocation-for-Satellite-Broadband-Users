from environment import environment
from RlAgent import PPo

env = environment(dataset= "optim_train_set.csv")
agent = PPo(env= env, nb_episodes= 25, alpha= 0.001, gamma= 0.5, epsilon= 1)

agent.train_agent()
agent.plots()