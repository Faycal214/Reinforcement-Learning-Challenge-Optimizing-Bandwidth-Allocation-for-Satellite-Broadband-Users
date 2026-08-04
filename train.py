import torch
import torch.nn as nn
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment import BandwidthAllocationEnv

class LSTMDemandForecaster(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        self.num_users = observation_space.shape[0]
        self.window_size = observation_space.shape[1]
        self.feat_dim = observation_space.shape[2]
        
        self.hidden_size = 32
        
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        linear_input_dim = self.num_users * self.hidden_size
        
        self.linear = nn.Sequential(
            nn.Linear(linear_input_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Reshape to process user time-series in parallel
        x = observations.view(batch_size * self.num_users, self.window_size, self.feat_dim)
        
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        
        x_flat = last_step.view(batch_size, self.num_users * self.hidden_size)
        return self.linear(x_flat)

def main():
    df = pd.read_csv('train_data.csv', parse_dates=['Date'], dayfirst=True)
    
    env = BandwidthAllocationEnv(df=df, num_users=10, window_size=5)
    
    policy_kwargs = dict(
        features_extractor_class=LSTMDemandForecaster,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01, # Forces the agent to explore different actions
        clip_range=0.2,
        verbose=1
    )
    
    print("Beginning Training Phase with LSTM Demand Forecaster...")
    # Increase to at least 500k to allow the LSTM to actually map the time-series
    model.learn(total_timesteps=500000)
    
    model.save("ppo_lstm_bandwidth_allocator")
    print("Model successfully saved.")

if __name__ == "__main__":
    main()