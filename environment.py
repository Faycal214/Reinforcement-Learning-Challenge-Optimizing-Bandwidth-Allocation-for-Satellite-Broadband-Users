import numpy as np
import pandas as pd

class environment:
    def __init__(self, dataset, time_steps = 288, total_bandwidth = 10000, num_users = 10, CIR = 1000):
        # Load dataset from CSV
        # Convert the 'Date' column to datetime format
        self.dataset = pd.read_csv(dataset)
        self.time_steps = time_steps  # Each row represents a time step
        self.num_users = num_users  # Set to 10 users
        self.total_bandwidth = total_bandwidth  # System capacity (10 Mbps)
        self.CIR = CIR  # Minimum bandwidth guarantee (1 Mbps or 1000 Kbps)

        # Initialize the environment's state and variables
        self.reset()
        self.state_size = 3

    def reset(self):
        """
        Reset the environment to the initial state and perform the initial allocation.
        """
        self.time_step = 0
        self.remaining_bandwidth = self.total_bandwidth
        self.allocated_bandwidth = [0] * self.num_users  # Start with no allocated bandwidth

        # Extract the initial requested bandwidth for all users from the dataset
        self.requested_bandwidth = self._get_requested_bandwidth(self.time_step)

        # Phase 1: Initial allocation
        self.initial_allocation()

        # Phase 2: MIRs set as initial allocation
        self.MIRs = self.allocated_bandwidth[:]

        # State: [MIRs, Requested Bandwidths, Allocated Bandwidths]
        self.state = self._get_state()
        return self.state

    def _get_requested_bandwidth(self, time_step):
        """
        Get the requested bandwidth for all users at the given time step from the dataset.

        Parameters:
        - time_step: The current time step in the simulation.

        Returns:
        - requested_bandwidths: A list of bandwidth requests for the current time step for all users.
        """
        a = time_step
        requested_bandwidths = []
        # Retrieve the requested bandwidth for each user from the dataset
        for _ in range(self.num_users) :
            requested_bandwidths.append(self.dataset.iloc[a][-1])
            a = a + self.time_steps

        return requested_bandwidths

    def _get_state(self):
        """
        Get the current state of the environment.

        Returns:
        - state: A list containing the state for all users.
        """
        state = (self.MIRs, self.requested_bandwidth, self.allocated_bandwidth)
        return state

    def initial_allocation(self):
        """
        Perform the initial allocation strategy (Phase 1) for all users.
        Ensure each user gets at least their CIR (or exactly their request if lower).
        """
        total_allocated = 0

        for i in range(self.num_users):
            if self.requested_bandwidth[i] >= self.CIR:
                allocated_bandwidth = self.CIR  # Allocate CIR if requested >= CIR
            else:
                allocated_bandwidth = self.requested_bandwidth[i]  # Allocate exact request if < CIR

            self.allocated_bandwidth[i] = allocated_bandwidth
            total_allocated += allocated_bandwidth

        self.remaining_bandwidth = self.total_bandwidth - total_allocated  # Update remaining bandwidth

    def step(self, actions):
        """
        Take actions to adjust the MIR for all users.

        Parameters:
        - actions: A list of actions for adjusting MIR for each user.

        Returns:
        - new_state: The updated state after the actions.
        - done: Whether the episode has finished.
        """
        """
        Adjust MIRs and allocate the remaining bandwidth (Phase 2)
        """
        # Phase 2: Adjust MIRs based on RL agent's actions
        sum_mirs = sum(self.MIRs)
        for i in range(self.num_users):
            mir = self.MIRs[i] + actions[i]
            if mir > self.remaining_bandwidth or sum_mirs > self.remaining_bandwidth:
                pass
            else :
                if mir < self.CIR :
                    self.MIRs[i] = self.CIR
                elif mir >= self.CIR :
                    self.MIRs[i] = mir
        
        # Get the updated state
        new_state = self._get_state()

        # Allocate remaining bandwidth based on the updated MIRs
        total_allocated = [self.allocate_bandwidth(i, self.requested_bandwidth[i], self.MIRs[i]) for i in range(self.num_users)]

        # Move to the next time step
        self.time_step += 1
        done = self.time_step >= self.time_steps

        # Calculate rewards based on the requested, allocated, and MIR values
        reward = self.calculate_rewards()

        # Calculate the average allocation ratio
        average_allocation_ratio = self.calculate_allocation_ratio()
        
        # Update the requested bandwidths for the next step
        if not done:
            self.requested_bandwidth = self._get_requested_bandwidth(self.time_step)

        return new_state, reward, done, average_allocation_ratio  # Returning


    def allocate_bandwidth(self, user_index, request_bandwidth, MIR):
        """
        Allocate bandwidth for a specific user based on their MIR and requested bandwidth.

        Parameters:
        - user_index: The user index to allocate bandwidth for.
        - request_bandwidth: The current requested bandwidth for the user.
        - MIR: The current maximum information rate for the user.

        Returns:
        - final_allocation: The final allocated bandwidth for the user (updated allocated bandwidth)
        """
        # Initial allocated bandwidth from Phase 1
        initial_allocation = self.allocated_bandwidth[user_index]

        # Calculate the remaining requested bandwidth
        remaining_requested = request_bandwidth - initial_allocation

        # Calculate the potential new allocation
        potential_allocation = initial_allocation + min(remaining_requested, MIR - initial_allocation)

        # Ensure that the total allocated bandwidth does not exceed 10,000 Kbps
        total_allocated = sum(self.allocated_bandwidth) + potential_allocation - initial_allocation
        # If the total allocated bandwidth exceeds the total bandwidth, return the allocated bandwidth instead
        if total_allocated > self.total_bandwidth or potential_allocation > MIR:
            return self.allocated_bandwidth[user_index]
        else :
            if request_bandwidth >= MIR :
                # Update the allocated bandwidth for the user
                self.allocated_bandwidth[user_index] = initial_allocation + (MIR - initial_allocation)
            else :
                self.allocated_bandwidth[user_index] = request_bandwidth
        return potential_allocation


    def calculate_allocation_ratio(self):
        """
        Calculate the average allocation ratio across all users.
        """
        """
        total_ratio = 0
        for i in range(self.num_users):
            if self.requested_bandwidth[i] >= self.MIRs[i]:
                ratio = self.MIRs[i] / self.requested_bandwidth[i]
            else:
                ratio = 1.0  # Fully satisfied

            total_ratio += ratio

        # Calculate average ratio
        sum_bandwidth_requested = sum(self.requested_bandwidth)
        average_allocation_ratio = total_ratio / sum_bandwidth_requested
        return average_allocation_ratio
        """
        allocation_ratio = [0] * self.num_users
        for i in range(self.num_users) :
            if self.requested_bandwidth[i] >= self.MIRs[i] :
                allocation_ratio[i] = self.MIRs[i] / self.total_bandwidth
            else :
                if self.requested_bandwidth[i] == 0 :
                    allocation_ratio[i] = self.allocated_bandwidth[i] / self.CIR
                else :
                    allocation_ratio[i] = self.allocated_bandwidth[i] / self.requested_bandwidth[i]
        return np.sum(allocation_ratio)

    def calculate_rewards(self):
        R_efficiency = 0.0
        abuse_counters = [0] * self.num_users
        penalty_coefficient = -0.5
        total_abuse_score = 0
        min_abuse_duration = 3
        theta = 0.2

        # Calculate R Efficiency and detect abuse
        for i in range(self.num_users):
            if self.requested_bandwidth[i] < self.MIRs[i]:
                R_efficiency += 1
            elif self.allocated_bandwidth[i] >= self.MIRs[i]:
                if self.requested_bandwidth[i] == 0 :
                    R_efficiency += self.allocated_bandwidth[i] / self.CIR
                else :
                    R_efficiency += self.MIRs[i] / self.requested_bandwidth[i]

            # Detect abusive behavior
            if self.requested_bandwidth[i] > self.MIRs[i] * (1 + theta):
                # Increment abuse counter
                abuse_counters[i] += 1
            else:
                # Reset abuse counter
                abuse_counters[i] = 0

            # If abuse counter reaches the minimum duration, count it as an abuse event
            if abuse_counters[i] >= min_abuse_duration:
                total_abuse_score += 1

        # Calculate P Over
        P_over = sum(max(0, requested - allocated) for requested, allocated in zip(self.requested_bandwidth, self.allocated_bandwidth))

        # Normalization for P_abusive
        # Use the number of users as the total possible abuse count to normalize
        possible_abuse_events = self.num_users
        P_abusive = penalty_coefficient * (total_abuse_score / possible_abuse_events) if possible_abuse_events > 0 else 0

        # Define weights for penalties
        alpha = 1.0  # Weight for P_over
        beta = 1.0   # Weight for P_abusive

        # Calculate total reward
        reward = R_efficiency - (alpha * P_over) - (beta * P_abusive)
        
        return reward
    
