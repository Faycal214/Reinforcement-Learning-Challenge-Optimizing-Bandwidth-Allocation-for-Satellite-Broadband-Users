import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from environment import environment

env = environment(dataset= "optim_train_set.csv")

class PPo:
    def __init__(self, env, nb_episodes, alpha, gamma, epsilon, eps_dec, eps_min) :
        self.env= env
        self.nb_episodes = nb_episodes
        self.alpha= alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.model = self.PolicyNetwork(self.env)
        self.optimizer = Adam(learning_rate= self.alpha)
        self.average_allocation_ratios = []

    class PolicyNetwork(tf.keras.Model) :
        def __init__(self, env) :
            super(PPo.PolicyNetwork, self).__init__()  # Proper inheritance
            self._input_ = Input(shape= (env.num_users, ))
            self.dense1= Dense(units= 128, activation = 'tanh')
            self.dense2 = Dense(units= 128, activation= 'tanh')
            self.logits = Dense(units=env.num_users)

        def __call__(self, state) :
            a = tf.convert_to_tensor(state, dtype= tf.float32)
            state = tf.expand_dims(a, axis= 0)
            # x = self._input_(state)
            x = self.dense1(state)
            x = self.dense2(x)
            return self.logits(x)
    
    def decrement_epsilon(self) :
        if self.epsilon > self.eps_min :
            self.epsilon -= self.eps_dec
        else :
            self.epsilon = self.eps_min
    
    def choose_action(self, state):
        random_number = np.random.rand()
        
        # Choisir une action aléatoire si epsilon est élevé
        if random_number < self.epsilon:
            actions = []
            for i in range(self.env.num_users):
                # Générer une action aléatoire autour de la demande moyenne de chaque utilisateur
                action = np.random.uniform(-150, 150)
                actions.append(action)
            return np.array(actions)
        else:
            # Passer l’état en entrée au modèle pour prédire l'action optimale
            logits = self.model(state)
            return logits.numpy().flatten()

    def compute_returns(self, rewards) :
        returns = []
        discounted_sum = 0
        for i, r in enumerate(reversed(rewards)) :
            # discounted_sum *= r + (self.gamma) ** (i)
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        # Normalize returns for better training stability
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train_agent(self):
        loss = 0
        all_ep_allocation_ratio = []
        for episode in range(self.nb_episodes):
            print(f"\nEpisode: {episode}")
            current_state = self.env.reset()
            current_state = current_state[0]  # Assuming state is a tuple, taking the first element
            rewards, states, actions, advantages, allocation_ratios = [], [], [], [], []
            done = False
            while not done:
                # Choose an action using epsilon-greedy approach
                action = self.choose_action(current_state)
                next_state, reward, done, average_allocation = self.env.step(action)
                next_state = next_state[0]
                # Store rewards, actions, and states
                rewards.append(reward)
                states.append(current_state)
                actions.append(action)
                allocation_ratios.append(average_allocation)
                # Move to next state
                current_state = next_state

            # After episode ends, compute returns (rewards-to-go)
            returns = self.compute_returns(rewards)
            allocation_all_episode = sum(allocation_ratios) / (self.env.num_users * self.env.time_steps)
            all_ep_allocation_ratio.append(allocation_all_episode)
            print(f"episode {episode} avrage allocation ratio {allocation_all_episode:.4f}")
            # Calculate advantages
            for state, action, reward, ret in zip(states, actions, rewards, returns):
                # Calculate the advantage (current reward - baseline)
                advantage = ret - self.model(state)  # This is just a simple example, you may want a value function
                advantages.append(advantage)

            # Train the model (optimize the policy)
            with tf.GradientTape() as tape:
                # Get the predicted logits from the model
                logits = self.model(states)

                # Calculate the loss (e.g., using the negative log likelihood of actions and advantages)
                loss = self.calculate_loss(logits, actions, advantages)
        
            # Calculate gradients and apply them
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Decay epsilon for exploration
            self.decrement_epsilon()

    def calculate_loss(self, logits, actions, advantages):
        # Calculate the probability of actions from the logits
        action_probs = tf.nn.softmax(logits)

        # Get the log probability of the taken actions
        action_log_probs = tf.reduce_sum(action_probs * actions, axis=1)

        # Calculate the loss using the surrogate objective
        surrogate_loss = -tf.reduce_mean(action_log_probs * advantages)

        # Include an entropy bonus to encourage exploration
        entropy_bonus = -0.01 * tf.reduce_mean(action_probs * tf.math.log(action_probs))
    
        # Total loss: policy loss + entropy loss
        return surrogate_loss + entropy_bonus
