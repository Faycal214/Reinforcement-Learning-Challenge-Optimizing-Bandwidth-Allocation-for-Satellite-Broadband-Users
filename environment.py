import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class BandwidthAllocationEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, num_users: int = 10, window_size: int = 5):
        super().__init__()
        self.df = df
        self.num_users = num_users
        self.window_size = window_size
        
        # System capacities
        self.total_capacity = 10000.0
        self.cir_max = 1000.0
        
        self.timestamps = sorted(self.df['Date'].unique())
        self.max_steps = len(self.timestamps)
        
        # Features per user per timestep: 
        # [Requested_BW, Phase1_Allocated, Abuse_Counter, Time_Sin, Time_Cos]
        self.feature_dim = 5
        
        # Observation space: 3D matrix for LSTM processing
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(self.num_users, self.window_size, self.feature_dim),
            dtype=np.float32
        )
        
        # Action space: Bounded logits to satisfy SB3 finite bound requirements
        self.action_space = spaces.Box(
            low=-5.0, 
            high=5.0, 
            shape=(self.num_users,), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.abuse_counters = np.zeros(self.num_users, dtype=np.float32)
        
        # Initialize historical rolling buffer
        self.history_buffer = np.zeros(
            (self.num_users, self.window_size, self.feature_dim), 
            dtype=np.float32
        )
        
        return self._get_obs(), {}

    def _get_obs(self):
        current_time = self.timestamps[self.current_step]
        step_df = self.df[self.df['Date'] == current_time]
        
        reqs = step_df['BW_REQUESTED'].values[:self.num_users]
        if len(reqs) < self.num_users:
            reqs = np.pad(reqs, (0, self.num_users - len(reqs)))
            
        phase1_alloc = np.minimum(reqs, self.cir_max)
        
        # Cyclic time encoding
        time_sin = np.sin(2 * np.pi * self.current_step / 288)
        time_cos = np.cos(2 * np.pi * self.current_step / 288)
        
        # Shift rolling window back by 1 step
        self.history_buffer[:, :-1, :] = self.history_buffer[:, 1:, :]
        
        # Update latest step
        for i in range(self.num_users):
            self.history_buffer[i, -1, 0] = reqs[i]
            self.history_buffer[i, -1, 1] = phase1_alloc[i]
            self.history_buffer[i, -1, 2] = self.abuse_counters[i]
            self.history_buffer[i, -1, 3] = time_sin
            self.history_buffer[i, -1, 4] = time_cos
            
        return np.copy(self.history_buffer)

    def step(self, action):
        current_time = self.timestamps[self.current_step]
        step_df = self.df[self.df['Date'] == current_time]
        reqs = step_df['BW_REQUESTED'].values[:self.num_users]
        if len(reqs) < self.num_users:
            reqs = np.pad(reqs, (0, self.num_users - len(reqs)))
            
        # 1. Phase 1 Constraint Guarantee (CIR)
        phase1_alloc = np.minimum(reqs, self.cir_max)
        rem_cap = max(0.0, self.total_capacity - np.sum(phase1_alloc))
        
        mirs = phase1_alloc.copy()
        budget = rem_cap
        
        # =================================================================
        # LAYER 1: OPERATIONS RESEARCH HEURISTIC (Constraint Satisfaction)
        # =================================================================
        # Deterministically rescue any user about to trigger a massive penalty.
        # This removes the impossible mathematical landscape for the RL agent.
        for i in range(self.num_users):
            if self.abuse_counters[i] == 2:
                # Calculate exactly what is needed to dodge the penalty
                target_mir = (reqs[i] / 1.2) + 0.001 
                deficit = target_mir - mirs[i]
                
                # If we have the budget and the deficit is logical, fill it
                if 0 < deficit <= budget:
                    mirs[i] += deficit
                    budget -= deficit

        # =================================================================
        # LAYER 2: REINFORCEMENT LEARNING (Stochastic Foresight)
        # =================================================================
        # The LSTM agent's raw actions are now treated as "Foresight Priority Scores".
        # We apply softmax only to the remaining budget after heuristics are satisfied.
        if budget > 0:
            exp_a = np.exp(action - np.max(action)) # Numerically stable softmax
            alpha = exp_a / exp_a.sum()
            mirs += (alpha * budget)
            
        # Cap allocations strictly at requested demand to avoid wasting bandwidth
        mirs = np.minimum(reqs, mirs)
        
        # 3. Efficiency Ratio Metric calculation
        r_i = np.where(reqs >= mirs, mirs / self.total_capacity, 1.0)
        r_i = np.where(reqs == 0, 1.0, r_i)
        efficiency_ratio = np.mean(r_i)
        
        # 4. Abuse Counter Tracking
        abuse_score = 0
        for i in range(self.num_users):
            if reqs[i] > mirs[i] * 1.2:
                self.abuse_counters[i] += 1
            else:
                self.abuse_counters[i] = 0
                
            if self.abuse_counters[i] >= 3:
                abuse_score += (self.abuse_counters[i] - 3)
                
        # 5. Hybrid Reward Formulation
        # With the OR layer handling the worst of the abuse, the agent is now 
        # free to focus heavily on optimizing the efficiency ratio.
        max_penalty = 15.0 
        normalized_abuse = min(abuse_score * 0.5, max_penalty) 
        reward = (efficiency_ratio * 20.0) - normalized_abuse
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps - 1
        
        info = {
            'efficiency_ratio': float(efficiency_ratio),
            'abuse_score': float(abuse_score)
        }
        
        return self._get_obs(), float(reward), terminated, False, info