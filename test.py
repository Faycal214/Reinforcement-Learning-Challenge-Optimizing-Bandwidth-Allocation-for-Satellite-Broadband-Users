from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import BandwidthAllocationEnv


DATA_PATH = "test_data.csv"
MODEL_PATH = "ppo_lstm_bandwidth_allocator"
WINDOW_SIZE = 5


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Put test_data.csv in the project root."
        )

    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=";")

    if df.empty:
        raise ValueError(f"{path} is empty.")
    return df


def main():
    print(f"Loading test data from {DATA_PATH}...")

    df = load_data(DATA_PATH)

    model_file = MODEL_PATH
    if not model_file.endswith(".zip"):
        model_file += ".zip"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing model: {model_file}. Run train.py first."
        )

    print(f"Loading model from {MODEL_PATH}...")

    raw_env = BandwidthAllocationEnv(
        df,
        window_size=WINDOW_SIZE,
        deterministic_shield=True,
    )
    env = DummyVecEnv([
        lambda: BandwidthAllocationEnv(
            df,
            window_size=WINDOW_SIZE,
            deterministic_shield=True,
        )
    ])

    model = PPO.load(MODEL_PATH, env=env)

    obs = env.reset()

    allocation_ratios = []
    total_allocations = []
    abuse_scores = []
    abuse_counters = []
    rewards = []
    over_penalties = []
    abusive_penalties = []
    capacity_violations = []

    all_allocated = []
    all_requests = []
    all_mirs = []
    all_phase1 = []

    done = False

    print()
    print("Running evaluation...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        info = infos[0]
        done = bool(dones[0])

        allocation_ratios.append(float(info["efficiency_ratio"]))
        total_allocations.append(float(info["total_allocation"]))
        abuse_scores.append(int(info["total_abuse_score"]))
        abuse_counters.append(info["abuse_counters"].copy())
        rewards.append(float(reward[0]))
        over_penalties.append(float(info["over_penalty"]))
        abusive_penalties.append(float(info["abusive_penalty"]))
        capacity_violations.append(float(info["capacity_violation"]))

        all_allocated.append(info["allocated"].copy())
        all_requests.append(info["requests"].copy())
        all_mirs.append(info["mir"].copy())
        all_phase1.append(info["phase1_allocation"].copy())

    env.close()

    allocation_ratios = np.asarray(allocation_ratios)
    total_allocations = np.asarray(total_allocations)
    abuse_scores = np.asarray(abuse_scores)
    rewards = np.asarray(rewards)
    over_penalties = np.asarray(over_penalties)
    abusive_penalties = np.asarray(abusive_penalties)
    capacity_violations = np.asarray(capacity_violations)

    allocated = np.asarray(all_allocated)
    requests = np.asarray(all_requests)
    mirs = np.asarray(all_mirs)
    phase1 = np.asarray(all_phase1)
    counters = np.asarray(abuse_counters)

    T = len(allocation_ratios)
    N = raw_env.num_users

    mean_allocation_ratio = float(allocation_ratios.mean())
    total_abuse_score = int(abuse_scores[-1]) if T else 0
    zero_abuse_steps = int(np.sum(abuse_scores == 0))
    mean_total_allocation = float(total_allocations.mean())
    max_total_allocation = float(total_allocations.max())
    capacity_violation_count = int(np.sum(capacity_violations > 1e-8))

    # Official request-normalized metric:
    # r_i,t = MIR/request when request >= MIR;
    #         1 otherwise.
    # We use final allocated bandwidth, which is equal to the request
    # whenever the request is below MIR.
    safe_request = np.where(requests > 0, requests, 1.0)
    official_ratios = np.where(
        requests <= 0,
        1.0,
        np.minimum(1.0, allocated / safe_request),
    )
    official_mean = float(official_ratios.mean())

    # Hard checks.
    demand_upper_bound_pass = bool(np.all(allocated <= requests + 1e-7))
    capacity_pass = bool(np.all(total_allocations <= 10_000.0 + 1e-7))

    # Phase 1 CIR rule:
    # request >= 1000 -> at least 1000 allocated in Phase 1;
    # request < 1000 -> exact request.
    expected_phase1 = np.minimum(requests, 1000.0)
    cir_pass = bool(np.all(np.abs(phase1 - expected_phase1) <= 1e-7))

    print()
    print("=" * 65)
    print("          V3 HYBRID PPO-LSTM + OR EVALUATION")
    print("=" * 65)
    print(f"Users Evaluated             : {N}")
    print(f"Time Steps Evaluated        : {T}")
    print(f"Mean Allocation Ratio       : {official_mean * 100:.2f}%")
    print(f"Total Abuse Score           : {total_abuse_score}")
    print(f"Zero-Abuse Timesteps        : {zero_abuse_steps}/{T}")
    print(f"Mean Total Allocation       : {mean_total_allocation:.2f} Kbps")
    print(f"Maximum Total Allocation    : {max_total_allocation:.2f} Kbps")
    print(f"Capacity Violations         : {capacity_violation_count}")
    print(f"Cumulative Reward           : {rewards.sum():.6f}")
    print(f"Mean Over Penalty           : {over_penalties.mean():.8f}")
    print(f"Mean Abusive Penalty        : {abusive_penalties.mean():.8f}")
    print("=" * 65)

    print()
    print("Running hard constraint checks...")
    print(
        f"Capacity constraint : {'PASS' if capacity_pass else 'FAIL'}"
    )
    print(
        f"Demand upper bound  : {'PASS' if demand_upper_bound_pass else 'FAIL'}"
    )
    print(
        f"CIR guarantee       : {'PASS' if cir_pass else 'FAIL'}"
    )

    # Diagnostics useful for determining whether abuse is structurally
    # difficult or caused by poor allocation decisions.
    phase1_remaining = 10_000.0 - phase1.sum(axis=1)
    target_mir = requests / 1.2
    rescue_gap = np.maximum(0.0, target_mir - np.maximum(1000.0, phase1))
    infeasible_rescue = rescue_gap.sum(axis=1) > phase1_remaining + 1e-7

    print()
    print("Feasibility diagnostics")
    print("-" * 65)
    print(
        f"Timesteps with mathematically insufficient residual "
        f"bandwidth for full 1.2x rescue : "
        f"{int(infeasible_rescue.sum())}/{T}"
    )
    print(
        f"Mean residual bandwidth after Phase 1 : "
        f"{phase1_remaining.mean():.2f} Kbps"
    )
    print(
        f"Maximum rescue gap                   : "
        f"{rescue_gap.sum(axis=1).max():.2f} Kbps"
    )

    # Plot 1: total capacity / allocation.
    x = np.arange(T)

    plt.figure(figsize=(12, 5))
    plt.plot(x, total_allocations, label="Total allocated")
    plt.axhline(10_000, linestyle="--", label="Capacity")
    plt.xlabel("Time step")
    plt.ylabel("Bandwidth (Kbps)")
    plt.title("Total Bandwidth Allocation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_bandwidth_plot.png", dpi=150)
    plt.close()
    print("\nBandwidth plot saved as 'evaluation_bandwidth_plot.png'.")

    # Plot 2: abuse counters.
    plt.figure(figsize=(12, 5))
    for i in range(N):
        plt.plot(x, counters[:, i], label=f"User {i + 1}")
    plt.axhline(3, linestyle="--", label="Abuse threshold")
    plt.xlabel("Time step")
    plt.ylabel("Consecutive abusive steps")
    plt.title("Abuse Counters")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("abuse_counter_plot.png", dpi=150)
    plt.close()
    print("Abuse counter plot saved as 'abuse_counter_plot.png'.")

    # Plot 3: mean request satisfaction.
    plt.figure(figsize=(12, 5))
    plt.plot(x, official_ratios.mean(axis=1))
    plt.axhline(0.95, linestyle="--", label="95% target")
    plt.xlabel("Time step")
    plt.ylabel("Allocation ratio")
    plt.title("Mean Allocation Ratio Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("allocation_ratio_plot.png", dpi=150)
    plt.close()
    print("Allocation-ratio plot saved as 'allocation_ratio_plot.png'.")


if __name__ == "__main__":
    main()