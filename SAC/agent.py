import numpy as np
import torch as T
import torch.nn.functional as F
from SAC.memory import ReplayBuffer
from SAC.networks import ActorNetwork, CriticNetwork, ValueNetwork
from torch.nn.utils import clip_grad_norm_

class Agent :
    def __init__(self, alpha, beta, input_dims, gamma, n_actions, 
                 max_size = 100000, tau = 0.005, layer1_size = 256, 
                 layer2_size = 256, batch_size = 512, reward_scale= 5) :
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        # self.max_action = env.action_space.high
        self.max_action = 100.0  # Match max_adjustment from ActorNetwork
        
        self.actor = ActorNetwork(alpha, input_dims, self.max_action, layer1_size, layer2_size, self.n_actions)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, layer1_size, layer2_size)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, layer1_size, layer2_size)
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size)
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size)
        
        self.scale = reward_scale
        self.update_network_parameters(tau = 1)
    
    def choose_action(self, state) :
        state = T.tensor(state).unsqueeze(0) # add batch dimension
        actions, _ = self.actor.sample_normal(state, reparameterize= False)
        
        return actions.cpu().detach().numpy().flatten() # remove batch dimension
    
    def remember(self, state, action, reward, next_state, done) :
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def update_network_parameters(self, tau = None) :
        if tau is None :
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        for name in value_state_dict :
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1 - tau) * target_value_state_dict[name].clone()
            
            self.target_value.load_state_dict(value_state_dict)
        
    def learn(self) :
        if self.memory.mem_cntr < self.batch_size :
            return 
        
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        
        state = T.tensor(state, dtype = T.float)
        next_state = T.tensor(next_state, dtype = T.float)
        action = T.tensor(action, dtype = T.float)
        reward = T.tensor(reward, dtype = T.float)
        done = T.tensor(done, dtype = T.bool)
        
        value = self.value.forward(state).view(-1)
        next_value = self.target_value.forward(next_state).view(-1)
        next_value[done] = 0.0
        
        actions, log_probs = self.actor.sample_normal(state, reparameterize= True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph = True)
        clip_grad_norm_(self.value.parameters(), 1.0)
        self.value.optimizer.step()
        
        actions, log_probs = self.actor.sample_normal(state, reparameterize= True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value_1 = T.min(q1_new_policy, q2_new_policy)
        critic_value_1 = critic_value_1.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor.optimizer.step()
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*next_value
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_loss_1 = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_loss_2 = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        
        critic_loss = critic_loss_1 + critic_loss_2
        critic_loss.backward()
        clip_grad_norm_(self.critic_1.parameters(), 1.0)
        clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()