from __future__ import annotations

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import BandwidthAllocationEnv


SEED = 42
DATA_PATH = "train_data.csv"
MODEL_PATH = "ppo_lstm_bandwidth_allocator"
BEST_MODEL_DIR = "./best_model"
CHECKPOINT_DIR = "./checkpoints"

WINDOW_SIZE = 5
TOTAL_TIMESTEPS = 500_000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMDemandForecaster(BaseFeaturesExtractor):
    """
    LSTM feature extractor over the rolling state window.

    The environment exposes:
        observation.shape = (window_size, feature_dim)

    The final hidden representation is passed to PPO's policy/value heads.
    """

    def __init__(
        self,
        observation_space,
        lstm_hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        feature_dim = observation_space.shape[-1]
        super().__init__(observation_space, features_dim=lstm_hidden_size)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(lstm_hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 preserves the 3-D Box shape for non-image observations.
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)

        observations = observations.float()
        output, _ = self.lstm(observations)
        features = output[:, -1, :]
        return self.norm(features)


def load_requests(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Put train_data.csv in the project root."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    return df


def make_env(df: pd.DataFrame):
    def _init():
        return BandwidthAllocationEnv(
            df,
            window_size=WINDOW_SIZE,
            deterministic_shield=True,
        )
    return _init


def main():
    set_seed(SEED)

    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df = load_requests(DATA_PATH)

    env = DummyVecEnv([make_env(df)])
    eval_env = DummyVecEnv([make_env(df)])

    policy_kwargs = dict(
        features_extractor_class=LSTMDemandForecaster,
        features_extractor_kwargs=dict(
            lstm_hidden_size=64,
            num_layers=1,
        ),
        net_arch=dict(
            pi=[128, 64],
            vf=[128, 64],
        ),
        activation_fn=nn.Tanh,
    )

    # PPO is retained as required by the challenge, while the LSTM
    # captures temporal demand structure.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=288,
        batch_size=72,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=SEED,
        device="auto",
        tensorboard_log="./tensorboard/",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=BEST_MODEL_DIR,
        eval_freq=288 * 5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=288 * 20,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_lstm_bandwidth",
    )

    print("=" * 70)
    print("TRAINING HYBRID PPO-LSTM + OR BANDWIDTH ALLOCATOR")
    print("=" * 70)
    print(f"Dataset       : {DATA_PATH}")
    print(f"Window        : {WINDOW_SIZE}")
    print(f"Timesteps     : {TOTAL_TIMESTEPS:,}")
    print("CIR           : 1000 Kbps")
    print("Capacity      : 10000 Kbps")
    print("beta          : 3")
    print("theta         : 0.2")
    print("delta_t_min   : 3")
    print("gamma_abusive : -0.5  [v3 specification]")
    print("=" * 70)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(MODEL_PATH)

    print()
    print("Model successfully saved to:")
    print(f"{MODEL_PATH}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()