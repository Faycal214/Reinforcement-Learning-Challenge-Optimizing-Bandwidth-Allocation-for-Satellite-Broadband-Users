
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class BandwidthAllocationEnv(gym.Env):
    """
    DevFest Algiers - Satellite Broadband Bandwidth Allocation.

    Implements the v3 challenge mechanics:
      Phase 1: CIR allocation.
      Phase 2: RL distributes only the residual bandwidth.
      Allocation ratio is request-normalized.
      Over-allocation penalty uses beta=3.
      Sustained abuse uses theta=0.2, minimum duration=3,
      gamma=-0.5 exactly as written in challenge1v3.md.

    IMPORTANT SPECIFICATION NOTE:
    The v3 text says both "MIR_i >= CIR" and
    "sum(MIR_i) <= Remaining Bandwidth". Since MIR is the TOTAL
    per-user cap used by the allocation equation, those two
    constraints are infeasible whenever Remaining < N*CIR.
    Therefore this implementation uses the operationally coherent
    interpretation: the RL action allocates ADDITIONAL MIR headroom
    above Phase-1 allocation, with sum(extra_MIR) <= Remaining.
    Final MIR = Phase1 allocation + extra_MIR.
    """

    metadata = {"render_modes": []}

    TOTAL_CAPACITY = 10_000.0
    CIR = 1_000.0
    BETA = 3.0
    THETA = 0.2
    DELTA_T_MIN = 3
    GAMMA_ABUSIVE = -0.5

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 5,
        episode_length: int | None = None,
        deterministic_shield: bool = True,
    ):
        super().__init__()

        self.requests = self._prepare_requests(data)
        self.num_users = self.requests.shape[1]
        self.total_steps = len(self.requests)
        self.window_size = int(window_size)
        self.episode_length = min(
            self.total_steps,
            episode_length if episode_length is not None else self.total_steps,
        )
        self.deterministic_shield = deterministic_shield

        # Features per user:
        # request, phase1 allocation, previous allocation,
        # previous MIR, abuse flag, abuse counter, previous extra MIR.
        self.features_per_user = 7
        self.global_features = 3  # remaining bandwidth + sin/cos time
        self.feature_dim = self.num_users * self.features_per_user + self.global_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.feature_dim),
            dtype=np.float32,
        )

        # Continuous priority/action vector. It is converted into a simplex
        # allocation over the residual bandwidth.
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_users,),
            dtype=np.float32,
        )

        self.t = 0
        self.start_t = 0
        self.end_t = self.episode_length
        self.abuse_counters = np.zeros(self.num_users, dtype=np.int32)
        self.abuse_scores = np.zeros(self.num_users, dtype=np.float64)
        self.prev_alloc = np.zeros(self.num_users, dtype=np.float64)
        self.prev_mir = np.full(self.num_users, self.CIR, dtype=np.float64)
        self.prev_extra = np.zeros(self.num_users, dtype=np.float64)

        # Evaluation history
        self.history = []

    @staticmethod
    def _prepare_requests(data: pd.DataFrame) -> np.ndarray:
        df = data.copy()

        did_col = next(
            (c for c in ["DID", "Device ID", "Device_ID", "device_id", "user_id"] if c in df.columns),
            None,
        )
        req_col = next(
            (
                c
                for c in [
                    "BW_REQUESTED",
                    "Bandwidth_Requested",
                    "Bandwidth Requested (Kbps)",
                    "request",
                    "requested_bandwidth",
                ]
                if c in df.columns
            ),
            None,
        )

        if req_col is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                raise ValueError("Could not find a bandwidth-request column.")
            req_col = numeric[-1]

        if did_col is None:
            # If the file is already wide, treat numeric columns as users.
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric) > 1:
                arr = df[numeric].to_numpy(dtype=np.float64)
                return np.maximum(arr, 0.0)

            # Otherwise a single request stream is treated as one user.
            return np.maximum(df[req_col].to_numpy(dtype=np.float64)[:, None], 0.0)

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.sort_values(["Timestamp", did_col])

        pivot = (
            df.pivot_table(
                index="Timestamp" if "Timestamp" in df.columns else df.index,
                columns=did_col,
                values=req_col,
                aggfunc="mean",
            )
            .sort_index()
        )

        pivot = pivot.ffill().fillna(0.0)
        return np.maximum(pivot.to_numpy(dtype=np.float64), 0.0)

    def _phase1(self, reqs):
        return np.minimum(reqs, self.CIR)

    def _remaining(self, phase1):
        return max(0.0, self.TOTAL_CAPACITY - float(np.sum(phase1)))

    def _time_features(self, t):
        period = max(self.total_steps, 1)
        angle = 2.0 * np.pi * (t % period) / period
        return np.array([np.sin(angle), np.cos(angle), self._remaining(self._phase1(self.requests[t])) / self.TOTAL_CAPACITY])

    def _build_feature_row(self, t):
        reqs = self.requests[t]
        phase1 = self._phase1(reqs)
        remaining = self._remaining(phase1)

        req_norm = reqs / self.TOTAL_CAPACITY
        p1_norm = phase1 / self.TOTAL_CAPACITY
        prev_alloc_norm = self.prev_alloc / self.TOTAL_CAPACITY
        prev_mir_norm = self.prev_mir / self.TOTAL_CAPACITY
        abuse_flag = (self.abuse_counters > 0).astype(np.float64)
        abuse_counter = np.clip(self.abuse_counters, 0, self.DELTA_T_MIN) / self.DELTA_T_MIN
        prev_extra_norm = self.prev_extra / self.TOTAL_CAPACITY

        per_user = np.stack(
            [
                req_norm,
                p1_norm,
                prev_alloc_norm,
                prev_mir_norm,
                abuse_flag,
                abuse_counter,
                prev_extra_norm,
            ],
            axis=1,
        ).reshape(-1)

        global_features = np.array(
            [
                remaining / self.TOTAL_CAPACITY,
                np.sin(2.0 * np.pi * (t % max(self.total_steps, 1)) / max(self.total_steps, 1)),
                np.cos(2.0 * np.pi * (t % max(self.total_steps, 1)) / max(self.total_steps, 1)),
            ]
        )

        return np.concatenate([per_user, global_features]).astype(np.float32)

    def _observation(self):
        rows = []
        for k in range(self.window_size):
            idx = max(0, self.t - self.window_size + 1 + k)
            idx = min(idx, self.total_steps - 1)
            rows.append(self._build_feature_row(idx))
        return np.stack(rows).astype(np.float32)

    @staticmethod
    def _safe_simplex(action):
        a = np.asarray(action, dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, 0.0, 1.0)
        s = float(a.sum())
        if s <= 1e-12:
            return np.ones_like(a) / len(a)
        return a / s

    def _allocate_residual(self, reqs, phase1, action):
        """
        Convert PPO action into extra MIR headroom.

        The residual is a shared budget. A user can never receive more
        than its outstanding request.
        """
        remaining = self._remaining(phase1)
        weights = self._safe_simplex(action)

        headroom = np.maximum(reqs - phase1, 0.0)
        extra = np.zeros(self.num_users, dtype=np.float64)

        # Deterministic OR safety layer:
        # if C_i=2, reserve the exact residual amount required to move
        # MIR_i to request_i / 1.2 whenever that rescue is feasible.
        # This acts before PPO receives the residual budget.
        at_risk = np.where(self.abuse_counters == self.DELTA_T_MIN - 1)[0]
        deficits = {
            int(i): max(
                0.0,
                reqs[i] / (1.0 + self.THETA) - max(self.CIR, phase1[i]),
            )
            for i in at_risk
        }

        for i in sorted(at_risk.tolist(), key=lambda j: deficits[j]):
            d = deficits[i]
            if d <= 1e-9:
                continue
            if d <= remaining:
                extra[i] += d
                remaining -= d
            else:
                # No complete rescue is feasible. Use the remaining
                # bandwidth to reduce the user's shortage, then stop
                # the rescue pass.
                extra[i] += remaining
                remaining = 0.0
                break

        if remaining <= 1e-9:
            mir = np.maximum(self.CIR, phase1 + extra)
            allocated = phase1 + np.minimum(headroom, extra)
            return mir, allocated, extra

        # PPO distributes only the residual bandwidth left after
        # deterministic safety reservations.
        weights = self._safe_simplex(action)
        residual_extra = np.minimum(weights * remaining, headroom - extra)
        extra += residual_extra

        # Redistribute overflow caused by saturated users.
        unused = remaining - float(residual_extra.sum())
        active = headroom - extra > 1e-9

        for _ in range(self.num_users):
            if unused <= 1e-9 or not np.any(active):
                break
            w = weights * active
            sw = float(w.sum())
            if sw <= 1e-12:
                w = active.astype(np.float64)
                sw = float(w.sum())

            add = np.minimum(unused * w / sw, headroom - extra)
            extra += add
            used = float(add.sum())
            unused -= used
            active = headroom - extra > 1e-9

        mir = np.maximum(self.CIR, phase1 + extra)
        allocated = phase1 + np.minimum(headroom, extra)

        return mir, allocated, extra

    def _apply_abuse_shield(self, reqs, phase1, mir, allocated, extra):
        """
        Deterministic preemptive shield.

        If a user is at C=2, try to raise MIR to request/1.2 using
        residual bandwidth. This is a deterministic safety layer,
        not an RL decision.
        """
        if not self.deterministic_shield:
            return mir, allocated, extra

        remaining = self._remaining(phase1)
        # Amount of residual budget not currently assigned.
        spare = max(0.0, remaining - float(extra.sum()))

        at_risk = np.where(self.abuse_counters == self.DELTA_T_MIN - 1)[0]
        if len(at_risk) == 0 or spare <= 1e-9:
            return mir, allocated, extra

        deficits = {}
        for i in at_risk:
            target = reqs[i] / (1.0 + self.THETA)
            deficits[i] = max(0.0, target - mir[i])

        # Cheapest complete rescues first.
        for i in sorted(at_risk.tolist(), key=lambda j: deficits[j]):
            d = deficits[i]
            if d <= 1e-9:
                continue
            if d <= spare:
                extra[i] += d
                mir[i] += d
                allocated[i] = phase1[i] + min(max(reqs[i] - phase1[i], 0.0), extra[i])
                spare -= d

        return mir, allocated, extra

    def _allocation_ratios(self, reqs, allocated):
        ratios = np.ones(self.num_users, dtype=np.float64)
        positive = reqs > 1e-12
        ratios[positive] = np.minimum(
            1.0,
            allocated[positive] / reqs[positive],
        )
        ratios[~positive] = 1.0
        return ratios

    def _update_abuse(self, reqs, mir):
        abusive = reqs > mir * (1.0 + self.THETA)

        self.abuse_counters = np.where(
            abusive,
            self.abuse_counters + 1,
            0,
        ).astype(np.int32)

        # S_i = D_i,j - Delta_t_min for each sustained episode.
        # Increment exactly when the current consecutive run exceeds 3.
        new_score = (self.abuse_counters > self.DELTA_T_MIN).astype(np.float64)
        self.abuse_scores += new_score

        return abusive

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.start_t = int(options.get("start_t", 0)) if options else 0
        self.start_t = max(0, min(self.start_t, self.total_steps - 1))
        self.end_t = min(self.total_steps, self.start_t + self.episode_length)
        self.t = self.start_t

        self.abuse_counters[:] = 0
        self.abuse_scores[:] = 0.0
        self.prev_alloc[:] = self._phase1(self.requests[self.t])
        self.prev_mir[:] = np.maximum(self.CIR, self.prev_alloc)
        self.prev_extra[:] = 0.0
        self.history = []

        return self._observation(), {}

    def step(self, action):
        if self.t >= self.end_t:
            raise RuntimeError("step() called after episode termination.")

        reqs = self.requests[self.t]
        phase1 = self._phase1(reqs)

        mir, allocated, extra = self._allocate_residual(reqs, phase1, action)
        mir, allocated, extra = self._apply_abuse_shield(
            reqs, phase1, mir, allocated, extra
        )

        # Numerical safety: never exceed capacity.
        total_alloc = float(allocated.sum())
        if total_alloc > self.TOTAL_CAPACITY + 1e-8:
            allocated *= self.TOTAL_CAPACITY / total_alloc

        ratios = self._allocation_ratios(reqs, allocated)
        efficiency = float(ratios.mean())

        over_penalty = self.BETA * max(
            0.0,
            float(allocated.sum()) - self.TOTAL_CAPACITY,
        ) / self.TOTAL_CAPACITY

        abusive = self._update_abuse(reqs, mir)
        s_total = int(self.abuse_scores.sum())

        # Exact v3 formula: gamma = -0.5.
        abusive_penalty = self.GAMMA_ABUSIVE * (
            s_total / max(1, self.num_users * self.total_steps)
        )

        reward = efficiency - over_penalty - abusive_penalty

        info = {
            "requests": reqs.copy(),
            "phase1_allocation": phase1.copy(),
            "remaining_bandwidth": self._remaining(phase1),
            "mir": mir.copy(),
            "extra_mir": extra.copy(),
            "allocated": allocated.copy(),
            "allocation_ratios": ratios.copy(),
            "efficiency_ratio": efficiency,
            "over_penalty": over_penalty,
            "abusive_penalty": abusive_penalty,
            "abusive": abusive.copy(),
            "abuse_counters": self.abuse_counters.copy(),
            "abuse_scores": self.abuse_scores.copy(),
            "total_abuse_score": s_total,
            "total_allocation": float(allocated.sum()),
            "capacity_violation": max(0.0, float(allocated.sum()) - self.TOTAL_CAPACITY),
        }
        self.history.append(info)

        self.prev_alloc = allocated.copy()
        self.prev_mir = mir.copy()
        self.prev_extra = extra.copy()

        self.t += 1

        terminated = self.t >= self.end_t
        truncated = False

        obs = (
            self._observation()
            if not terminated
            else self._observation()
        )

        return obs, float(reward), terminated, truncated, info
