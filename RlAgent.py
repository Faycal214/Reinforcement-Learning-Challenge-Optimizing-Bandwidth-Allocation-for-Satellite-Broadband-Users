import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import json

class PPo :
    def __init__(self, env, nb_episodes, alpha, gamma, epsilon) :
        self.env= env
        self.nb_episodes = nb_episodes
        self.alpha= alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = self.PolicyNetwork(self.env)
        self.optimizer = Adam(learning_rate= self.alpha)
        self.rewards_all_ep= []
        self.allocated_bandwidth_all_ep = []  # To store allocated_bandwidth across episodes
        self.requested_bandwidth_all_ep = []  # To store MIRs across episodes
        self.best_average_allocation_ratio = 0
        self.episode_of_best_average_allocation_ratio = 0


    class PolicyNetwork(tf.keras.Model) :
        def __init__(self, env) :
            super(PPo.PolicyNetwork, self).__init__()  # Proper inheritance
            self.dense1= Dense(units= 24, activation = 'relu')
            self.dense2 = Dense(units= 24, activation= 'relu')
            self.logits = Dense(units=env.num_users)

        def __call__(self, state) :
            state = tf.convert_to_tensor(state, dtype= tf.float32)
            state = tf.expand_dims(state, axis= 0)
            # state = tf.reshape(state, [1, -1]) #Reshape the state
            x = self.dense1(state)
            x = self.dense2(x)
            return self.logits(x)
    
    
    def calculate_mean_demands(self, demands):
        """
        Calcule la demande moyenne pour chaque client en utilisant les données historiques
        """
        mean_demands = sum(demands) / len(demands)
        return mean_demands
    
    def choose_action(self, state):
        random_number = np.random.rand()
        
        # Choisir une action aléatoire si epsilon est élevé
        if random_number < self.epsilon:
            actions = []
            mean_demands = self.calculate_mean_demands(state[1])
            for i in range(self.env.num_users):
                # Générer une action aléatoire autour de la demande moyenne de chaque utilisateur
                action = np.random.uniform(-0.5 * mean_demands, 0.5 * mean_demands)
                actions.append(action)
            return np.array(actions)
        else:
            # Passer l’état en entrée au modèle pour prédire l'action optimale
            logits = self.model(state[0])
            return logits.numpy().flatten()

    def compute_returns(self, rewards) :
        returns = []
        discounted_sum = 0
        for i, r in enumerate(reversed(rewards)) :
            # discounted_sum *= r + (self.gamma) ** (i)
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns

    def train_agent(self) :
        for episode in range(self.nb_episodes) :
            print(f"\nEpisode : {episode}")
            current_state = self.env.reset()
            rewards, states, action = [], [], []
            allocated_bandwidth = []
            requested_bandwidth = []
            done = False
            while not done :
                # choose an action with epsilon-greedy approach
                actions = self.choose_action(current_state)
                action.append(actions)
                # apply the action in the environment and observ the next state, rawards
                next_state, reward, done, allocated_bandwidth_step, requested_bandwidth_step, average_allocation_ratio = self.env.step(actions)
                # save the state, action
                states.append(next_state)
                rewards.append(reward)
                allocated_bandwidth.append(allocated_bandwidth_step)
                requested_bandwidth.append(requested_bandwidth_step)
                # update the state
                current_state = next_state
            
            self.allocated_bandwidth_all_ep.append(allocated_bandwidth) # pour la visualisation
            self.requested_bandwidth_all_ep.append(requested_bandwidth) # aussi pour la visualisation

            # calculer les retours G_t
            returns = self.compute_returns(rewards)
            total_rewards = sum(returns)
            self.rewards_all_ep.append(total_rewards)

            if episode > 10 :
                self.epsilon = self.epsilon * 0.9

            if average_allocation_ratio >= self.best_average_allocation_ratio :
                self.best_average_allocation_ratio = average_allocation_ratio
                self.episode_of_best_average_allocation_ratio = episode

            # Once all episodes are done, save to JSON files
            # self.save_to_json()

            # update the policy
            with tf.GradientTape() as tape:
                loss = 0
                for i in range(len(states)):
                    state = tf.convert_to_tensor(states[i][0], dtype=tf.float32)
                    logits = self.model(state)

                    # Calcul des probabilités d'action
                    action_proba = tf.nn.softmax(logits)

                    # Représentation one-hot de l'action, il faut s'assurer que la forme correspond
                    action_one_hot = tf.one_hot(action[i], self.env.num_users)

                    # Ajuster la forme de action_one_hot pour qu'elle corresponde à celle de action_proba
                    action_one_hot = tf.expand_dims(action_one_hot, axis=0)  # Ajouter une dimension batch

                    # Multiplier les deux tenseurs
                    selected_action_proba = tf.reduce_sum(action_proba * action_one_hot)

                    # Calcul de la perte (log-likelihood)
                    loss -= tf.math.log(selected_action_proba) * rewards[i]
            # calculate the gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    def plots(self):
        # Extract allocated and requested bandwidth for the first user only
        allocated_user = [self.allocated_bandwidth_all_ep[-1][j][0] for j in range(len(self.allocated_bandwidth_all_ep[-1]))]
        requested_user = [self.requested_bandwidth_all_ep[-1][j][0] for j in range(len(self.requested_bandwidth_all_ep[-1]))]
        
        # Plotting for the first user
        plt.figure(figsize=(14, 12))
        
        # Plot allocated and requested bandwidth on the same plot for comparison
        plt.plot(allocated_user, label="Allocated Bandwidth", color="blue")
        plt.plot(requested_user, label="Requested Bandwidth", color="orange", linestyle="--")
        
        # Add title and labels
        plt.title("User 1 Bandwidth Allocation vs. Request")
        plt.xlabel("Time Steps")
        plt.ylabel("Bandwidth (Mbps)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("user1_bandwidth_variability_visualization.png")
        plt.show()
