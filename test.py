import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import BandwidthAllocationEnv
from train import LSTMDemandForecaster 


def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(file_path, sep=';')
        
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    else:
        raise KeyError(f"Could not locate 'Date' column in {file_path}. Columns found: {list(df.columns)}")
        
    return df


def evaluate_agent(data_path: str = 'test_data.csv', model_path: str = 'ppo_lstm_bandwidth_allocator'):
    print(f"Loading test data from {data_path}...")
    df = load_dataset(data_path)
    
    env = BandwidthAllocationEnv(df=df, num_users=10, window_size=5)
    
    print(f"Loading saved model weights from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    obs, info = env.reset()
    done = False
    
    total_reward = 0.0
    ratios_history = []
    abuse_history = []
    user_0_requested = []
    user_0_allocated = []
    
    step_count = 0
    
    print("\nRunning Evaluation Loop...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        # Track User 0 demand vs allocated capacity
        current_time = env.timestamps[env.current_step]
        step_df = env.df[env.df['Date'] == current_time]
        reqs = step_df['BW_REQUESTED'].values[:env.num_users]
        if len(reqs) < env.num_users:
            reqs = np.pad(reqs, (0, env.num_users - len(reqs)))
            
        phase1_alloc = np.minimum(reqs, env.cir_max)
        rem_cap = max(0.0, env.total_capacity - np.sum(phase1_alloc))
        
        exp_a = np.exp(action - np.max(action))
        alpha = exp_a / exp_a.sum()
        mirs = phase1_alloc + (alpha * rem_cap)
        actual_allocs = np.minimum(reqs, mirs)
        
        user_0_requested.append(reqs[0])
        user_0_allocated.append(actual_allocs[0])
        
        # Advance environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Log metric values from info dict
        total_reward += reward
        ratios_history.append(info['efficiency_ratio'])
        abuse_history.append(info['abuse_score'])
        step_count += 1

    # Final Score Summary
    mean_ratio = np.mean(ratios_history) * 100.0
    total_abuse = np.sum(abuse_history)

    print("\n" + "=" * 55)
    print("           HACKATHON EVALUATION METRICS SUMMARY")
    print("=" * 55)
    print(f" Total Steps Evaluated       : {step_count}")
    print(f" Mean Allocation Ratio (R)  : {mean_ratio:.2f}%  <-- YOUR RATIO")
    print(f" Total Abuse Score (S_total): {total_abuse:.0f}")
    print(f" Cumulative Episode Reward  : {total_reward:.2f}")
    print("=" * 55)

    plot_results(user_0_requested, user_0_allocated)


def plot_results(requested, allocated, num_steps: int = 250):
    plt.figure(figsize=(14, 6))
    plt.plot(requested[:num_steps], label='Requested Bandwidth (Kbps)', color='#d9534f', linestyle='--', linewidth=1.5)
    plt.plot(allocated[:num_steps], label='Allocated Bandwidth (Kbps)', color='#0275d8', linewidth=2.0)
    
    plt.title(f'PPO-LSTM Preemptive Demand Matching (First {num_steps} Timesteps)', fontsize=14, fontweight='bold')
    plt.xlabel('Timestep Index', fontsize=12)
    plt.ylabel('Bandwidth (Kbps)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('evaluation_bandwidth_plot.png', dpi=300)
    print("\nPlot saved as 'evaluation_bandwidth_plot.png'.")
    plt.show()

if __name__ == "__main__":
    evaluate_agent()