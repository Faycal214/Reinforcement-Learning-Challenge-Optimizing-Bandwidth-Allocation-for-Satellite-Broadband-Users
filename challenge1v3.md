# Hacker Guide
The DevFest Algiers hackathon is primarily associated with university students from different technical backgrounds: programmers, ethical hackers, and software coders. It is a place where like-minded people come together to "hack" and craft realistic technical solutions.

The goal of this year’s edition of Devfest Algiers Hackathon is to come up with a solution related to the theme of the edition during the event and to create functional software products by the end of the event.

The hackathon has two phases, where the online lasts for 2 days, and the onsite phase for further 2 days to the teams that qualify. The hackathon also includes a prize for the winning teams.
# First challenge: **Reinforcement Learning Challenge: Optimizing Bandwidth Allocation for Satellite Broadband Users**

---

## **Introduction**

Welcome to the **Optimizing Bandwidth Allocation for Satellite Broadband Users** challenge!

In today's digital age, providing high-speed internet over shared satellite bandwidth is a significant challenge. Operators must balance delivering an optimal customer experience with managing operational costs and system constraints. This challenge invites you to develop a Reinforcement Learning (RL) solution that dynamically allocates bandwidth among users, enhancing their Quality of Experience (QoE) while adhering to specified operational constraints.

---

## **Challenge Overview**

### **Objective**

The **primary goal** is to **maximize the Average Allocation Ratio across all users over a 24-hour period**. The Allocation Ratio is a measure of **how effectively we satisfy each user's bandwidth request** relative to the system's capacity and constraints.

### **Constraints and Rules**

1. **Minimum Bandwidth Guarantee (Committed Information Rate - CIR):**
   - Each user must receive at least **1 Mbps** (1000 Kbps) at all times if they request at least **1 Mbps**.
   - If a user requests less than **1 Mbps**, they receive exactly what they request.
   - **CIR is guaranteed in the initial allocation phase and cannot be violated by the RL agent.**

2. **Maximum System Capacity:**
   - Total bandwidth allocated across all users must **not exceed 10 Mbps** (10,000 Kbps).

3. **User-Specific Maximum Information Rate (MIR):**
   - The RL agent adjusts the MIRs for each user, which should be:
     - **MIR $_{i}$ ≥ CIR (1000 Kbps)**
     - **MIR $_{i}$ ≤ Remaining Bandwidth**

4. **Sum of MIRs for all users:**
   - $\sum_{i=1}^{N} \text{MIR}_i \leq \text{Remaining Bandwidth}$
   - **Remaining Bandwidth** is the total bandwidth left after the CIR allocations.

5. **Prevent Sustained Abusive Usage:**
   - Penalize users who **consistently request bandwidth significantly exceeding their MIR by more than 20%** over extended periods.

---

## **Challenge Components**

1. **Environment Setup**
2. **Initial Allocation Strategy**
3. **Reinforcement Learning Model Development**
4. **Reward and Penalty Mechanism**
5. **Submission Guidelines**
6. **Evaluation Criteria**

---

### **1. Environment Setup**

#### **Scenario Parameters**

- **Total Bandwidth (System Capacity):** 10 Mbps (10,000 Kbps)
- **Number of Users (Clients):** 10
- **Minimum Bandwidth per User (CIR):** 1 Mbps (1000 Kbps)
- **Time Frame:** 24 hours, with time steps every 5 minutes (total of 288 time steps for each client)

#### **Dataset**

You will be provided with synthetic data for training (`train_data.csv`) simulating bandwidth requests from 10 users, categorized into **4 profiles (Low and Regular Usage, Medium Usage, High and Intensive Usage, and Business Clients)** over a 24-hour period.

**Dataset Format:**

| Device ID (DID) | Timestamp            | Bandwidth Requested (Kbps) |
|-----------------|----------------------|----------------------------|
| 1               | 2024-10-12 00:00:00  | 1035                       |
| 1               | 2024-10-12 00:05:00  | 950                        |
| ...             | ...                  | ...                        |

**Dataset Link:**
[click here](https://drive.google.com/drive/folders/1HcHDarR1ZX-R_QTDQwjhPcMUtS5tYWN-?usp=sharing)

#### **State Space**

At each time step, we suggest the state includes:

- **For each user:**
  - **Current MIR:** Maximum bandwidth limit set for the user (in Kbps).
  - **Bandwidth Requested:** Bandwidth the user wants at the current time (in Kbps).
  - **Bandwidth Allocated:** Bandwidth assigned to the user at the previous time step (in Kbps).
  - **Abusive Usage Indicator:** Binary flag indicating if the user is currently in an abusive usage episode.
  - **Time of Day:** Encoded appropriately to capture usage patterns over time.

#### **Action Space**

- **Adjust MIRs:** Increase or decrease the MIR for each user within allowable limits (MIR ≥ 1000 Kbps, sum of MIRs ≤ Remaining Bandwidth).

---

### **2. Initial Allocation Strategy**

Implement the following allocation strategy at each time step **before** the RL agent makes adjustments:

#### **Phase 1: Ensure Minimum Bandwidth (CIR)**

- **Allocate 1 Mbps (1000 Kbps)** to every user if their requested bandwidth is at least **1 Mbps**.
- If a user requests less than **1 Mbps**, **allocate exactly what they request**.

**Note:** The CIR is **guaranteed** in this phase, and cannot be violated by the RL agent.

#### **Phase 2: RL Agent Optimizes Remaining Bandwidth**

- **Calculate Remaining Bandwidth:**

  $
  \text{Remaining Bandwidth} = 10,000 \text{ Kbps} - \sum_{i=1}^{N} \text{Allocated}_{i,t}^{\text{Phase 1}}
  $

- **RL Agent's Role:**

  - The RL agent uses the **remaining bandwidth** to adjust **Maximum Information Rates (MIRs)** for users.
  - The agent decides how to distribute the remaining bandwidth among users to **maximize the average allocation ratio** while adhering to constraints.

- **Constraints for RL Agent:**

  - For each user $ i $:

    - **MIR $_{i,t}$ ≥ CIR (1000 Kbps)**
    - **MIR $_{i,t}$ ≤ Remaining Bandwidth**

  - **Sum of MIRs across all users must not exceed the Remaining Bandwidth.**

- **Allocation After RL Adjustments:**

  - For each user, **Allocated Bandwidth** is:

    $
    \text{Allocated}_{i,t} = \text{Allocated}_{i,t}^{\text{Phase 1}} + \min \left( \text{Bandwidth Requested}_{i,t} - \text{Allocated}_{i,t}^{\text{Phase 1}},\ \text{MIR}_{i,t} - \text{Allocated}_{i,t}^{\text{Phase 1}} \right)
    $

  - Ensure that **total allocated bandwidth** does not exceed **10,000 Kbps**.

**Note:** The RL agent works **only on the remaining bandwidth** after the initial allocation that guarantees the CIR for each user.

---

### **3. Reinforcement Learning Model Development**

Develop an RL agent that adjusts each user's MIR to optimize the allocation of the **remaining bandwidth** after the initial allocation.

#### **Requirements**

- **Implementation Correctness (20 points):**
  - Correctly implement the environment and initial allocation strategy.
  - Accurately model the state and action spaces.
  - Ensure that the RL agent operates on the remaining bandwidth after CIR allocations.

- **Design Decisions (20 points):**
  - Appropriately choose and justify the RL algorithm used (e.g., DQN, PPO).
  - Define state and action representations effectively.
  - Demonstrate creativity and innovation in your approach.

#### **Steps**

1. **Define the Environment:**
   - Implement the initial allocation strategy within your environment.
   - Include state and action spaces as defined.
   - Ensure that the agent operates on the remaining bandwidth after Phase 1.

2. **Design the RL Agent:**
   - Ensure the agent learns to adjust MIRs based on the state.
   - The MIR adjustments should be within the allowable range (MIR ≥ 1000 Kbps, sum of MIRs ≤ Remaining Bandwidth).

3. **Implement Reward Function:**
   - Design the reward function as per the specifications, accurately capturing the objectives and penalties.
   - The reward should reflect the agent's effectiveness in allocating the remaining bandwidth.

4. **Training:**
   - Train your agent using the provided training dataset.

5. **Testing:**
   - Evaluate your agent on the testing dataset, unseen during training (`test_data.csv`).

---

### **4. Reward and Penalty Mechanism**

#### **4.1. Objective: Maximize Average Allocation Ratio**

At each time step, for each user $ i $:

- **Goal:** Maximize the average allocation ratio over all users and time steps.

#### **Allocation Ratio Calculation**

The **Allocation Ratio** is calculated differently based on the user's requested bandwidth relative to their MIR and the system's maximum capacity (**10 Mbps**).

**Scenarios:**

**a. If Requested Bandwidth ≥ MIR:**

- **Allocated Bandwidth:**

  $
  \text{Allocated}_{i,t+1} = \text{MIR}_{i,t}
  $

- **Allocation Ratio:**

  $
  \text{Allocation Ratio}_{i,t} = \frac{\text{MIR}_{i,t}}{\text{Bandwidth Requested}_{i,t}}
  $

**b. If Requested Bandwidth < MIR:**

- **Allocated Bandwidth:**

  $
  \text{Allocated}_{i,t+1} = \text{Bandwidth Requested}_{i,t}
  $

- **Allocation Ratio:**

  $
  \text{Allocation Ratio}_{i,t} = \frac{\text{Allocated}_{i,t}}{\text{Bandwidth Requested}_{i,t}} = 1
  $

**Considerations:**

- **Total Allocations Must Not Exceed System Capacity:**
  
  $
  \sum_{i=1}^{N} \text{Allocated}_{i,t} \leq 10,000 \text{ Kbps}
  $

- **Allocations Must Not Exceed MIRs:**

  $
  \text{Allocated}_{i,t} \leq \text{MIR}_{i,t}
  $

#### **4.2. Reward Function Components**

**Total Reward at Each Time Step ($ R_t $):**

$
R_t = R_{\text{efficiency}} - P_{\text{over}} - P_{\text{abusive}}
$

---

Below is a detailed description of the reward function:

### **a. Bandwidth Allocation Efficiency Reward ($ R_{\text{efficiency}} $)**

We calculate $ R_{\text{efficiency}} $ as the **average efficiency across all users**.

$
R_{\text{efficiency}} = \frac{1}{N} \sum_{i=1}^{N} r_{i,t}
$

**Where $ r_{i,t} $ is the efficiency for user $ i $ at time $ t $, calculated as follows:**

- **If Requested Bandwidth ≥ MIR:**

  $
  r_{i,t} = \frac{\text{MIR}_{i,t}}{\text{Bandwidth Requested}_{i,t}}
  $

- **If Requested Bandwidth < MIR:**

  $
  r_{i,t} = \frac{\text{Allocated}_{i,t}}{\text{Bandwidth Requested}_{i,t}} = 1
  $


**Interpretation:**

- **When Requested Bandwidth ≥ MIR:**

  - The user's satisfaction is measured relative to the requested bandwidth.

- **When Requested Bandwidth < MIR:**
  - The user's request is fully satisfied, achieving $ r_{i,t} = 1 $.

### **b. Penalty for Over-Allocation of Total Bandwidth ($ P_{\text{over}} $)**

We calculate $ P_{\text{over}} $ based on the **total bandwidth over-allocation**, as a proportion of the maximum capacity.

$
P_{\text{over}} = \beta \times \frac{\max\left(0,\ \sum_{i=1}^{N} \text{Allocated}_{i,t} - 10,000 \right)}{10,000}
$

- **Penalty Coefficient:** $ \beta = 3 $ (Can be finetuned if needed)

**Interpretation:**

- Penalizes the agent for exceeding the system's capacity.
- Ensures adherence to constraints by discouraging over-allocation of total bandwidth.
- The penalty increases proportionally to the extent of the over-allocation.

### **c. Penalty for Sustained Abusive Usage ($ P_{\text{abusive}} $)**

We calculate $ P_{\text{abusive}} $ based on the **aggregate abuse score across all users**, normalized by the total possible abuse score.

**Parameters:**

- **Threshold Exceedance ($ \theta $)**: 20% (i.e., $ \theta = 0.2 $)
- **Minimum Duration ($ \Delta t_{\text{min}} $)**: 3 consecutive time steps
- **Penalty Coefficient:** $ \gamma = -0.5 $ (Can be fine-tuned if needed)

**Abusive Usage Detection Procedure:**

1. **For each user $ i $ and time step $ t $:**

   - **Condition:**

     $
     \text{Bandwidth Requested}_{i,t} > \text{MIR}_{i,t} \times (1 + \theta)
     $

   - **Abuse Counter ($ C_i $):**

     - **If condition is met:** Increment $ C_i $.
     - **If condition is not met:** Reset $ C_i = 0 $.

2. **Abusive Episode:**

   - An abusive episode occurs when $ C_i \geq \Delta t_{\text{min}} $.
   - **Duration ($ D_{i,j} $)**: The total consecutive time steps where the abuse condition persists **beyond** $ \Delta t_{\text{min}} $.

3. **Calculate Abuse Score for User $ i $:**

   $
   S_i = \sum_{j} (D_{i,j} - \Delta t_{\text{min}})
   $

4. **Aggregate Abuse Scores Across All Users:**

   $
   S_{\text{total}} = \sum_{i=1}^{N} S_i
   $

5. **Calculate Penalty:**

   $
   P_{\text{abusive}} = \gamma \times \frac{S_{\text{total}}}{N \times T}
   $

   - $ N $: Total number of users
   - $ T $: Total number of time steps

**Interpretation:**

- Penalizes the agent for sustained abusive usage across the user base.
- Normalizing the total abuse score ensures the penalty is proportionate to the extent of abuse among all users over time.
- Encourages the agent to manage and prevent sustained overuse of bandwidth by multiple users.
- Short-term spikes are not penalized, allowing for normal fluctuations.

---

### **How Penalties Ensure Convergence**

- **Penalties guide the RL agent towards desirable behavior by reducing the reward when constraints are violated across the user base.**

  - **Over-Allocation Penalty:** Ensures the agent manages total allocations to stay within system capacity.

  - **Abusive Usage Penalty:** Discourages the agent from allowing sustained abusive behavior among users.

- **Convergence:**

  - By optimizing the reward function, which balances maximizing the average allocation efficiency and minimizing penalties, the agent learns policies that meet objectives while adhering to constraints.
  
  - The agent's goal is to maximize the total reward over time, considering the impact of decisions on all users, leading to policies that benefit overall system performance.

---

### **Clarifications on Using the Model with Initial Allocation**

- **Initial Allocation (Phase 1):** Ensures that every user receives at least their CIR or exact request if less than 1 Mbps. The RL agent cannot affect these allocations.

- **RL Agent Adjustments (Phase 2):** The RL agent works only on the **remaining bandwidth** to adjust MIRs for each user, optimizing the allocation of additional bandwidth.

- **Reward Calculation:**

  - The reward is calculated after both phases, considering the final allocations to users.

  - The agent's actions (MIR adjustments) directly impact the rewards through their influence on the allocation ratios and penalties.

- **Constraints Management:**

  - The agent must ensure that its MIR adjustments do not cause the total allocated bandwidth to exceed the remaining bandwidth or system capacity.

---

### **5. Submission Guidelines**

#### **Code Submission**

- **Format:** Submit all code files necessary to run your solution.

- **Documentation:**
  - **Code Comments:** Ensure your code is well-commented for clarity.
  - **README File:** Provide instructions on how to set up and run your code.

- **Dependencies:**
  - List all libraries and dependencies required to run your code.

#### **Report Submission**

- **Format:** PDF, maximum of **5 pages**.

- **Content Structure:**
  1. **Introduction:**
     - Briefly describe your understanding of the problem and objectives.
  2. **Methodology:**
     - **Model Architecture (10 points):**
       - Describe your RL agent, including algorithms and architectures used.
       - Justify your choices.
     - **State and Action Spaces (5 points):**
       - Detail how you defined the state and action spaces.
     - **Reward Function Design (5 points):**
       - Explain how you implemented the reward and penalty mechanisms.
       - Include any additional considerations or adjustments made.
  3. **Results:**
     - Present your evaluation metrics.
     - Show relevant graphs or tables illustrating performance over time (e.g., allocation ratios, penalties).
  4. **Discussion:**
     - Analyze the results.
     - Discuss challenges faced and how you addressed them.
     - Reflect on the effectiveness of your approach and any limitations.
  5. **Conclusion:**
     - Summarize your findings.
     - Suggest potential improvements or future work.

#### **Submission Platform**

- Submissions should be made through the link: [Submission Form](https://forms.gle/MYG4nqnfttvvtSZCA)

---

### **6. Evaluation Criteria**

Your submission will be evaluated on the following criteria:

#### **Total Points Available: 100**

---

#### **A. Implementation and Correctness (40 points)**

1. **Environment and Initial Allocation Strategy (10 points):**
   - Correct implementation of the environment dynamics and initial allocation strategy.
   - Ensuring that the RL agent operates on the remaining bandwidth after initial allocation.

2. **State and Action Space Definition (10 points):**
   - Effective and accurate representation of states and actions.
   - Reflecting the focus on the remaining bandwidth.

3. **Reward Function Implementation (20 points):**
   - Correct calculation of:
     - $ R_{\text{efficiency}} $ (10 points)
     - $ P_{\text{over}} $ (5 points)
     - $ P_{\text{abusive}} $ (5 points)

---

#### **B. Model Design and Decisions (20 points)**

1. **Choice of RL Algorithm (10 points):**
   - Appropriateness of the algorithm for the problem.
   - Justification for the choice.

2. **Innovativeness and Creativity (10 points):**
   - Innovative approaches to problem-solving.
   - Effective handling of constraints and optimization.

---

#### **C. Performance Metrics (30 points)**

1. **Average Allocation Ratio (20 points):**

   $
   \text{Average Allocation Ratio} = \frac{1}{N \times T} \sum_{t=1}^{T} \sum_{i=1}^{N} \left( \text{Allocation Ratio}_{i,t} \right)
   $

   - **Scoring:**

     | Allocation Ratio (%) | Points |
     |----------------------|--------|
     | ≥ 95%                | 20     |
     | 90% - 94%            | 15     |
     | 85% - 89%            | 10     |
     | 80% - 84%            | 5      |
     | < 80%                | 0      |

2. **Penalty Minimization (10 points):**

   - **Preventing Over-Allocation (5 points):**
     - No over-allocation occurrences → **5 Points**
     - Any over-allocation occurrences → **0 Points**

   - **Managing Sustained Abusive Usage (5 points):**
     - Based on the total abuse score:
       - $ S_{\text{total}} = 0 $ → **5 Points**
       - $ S_{\text{total}} > 0 $ → **0 Points**

---

#### **D. Report Quality (10 points)**

1. **Clarity and Organization (5 points):**
   - Well-structured report with clear explanations.

2. **Depth of Analysis (5 points):**
   - Insightful discussion of results and challenges.
   - Reflection on limitations and potential improvements.

---

### **Total Points: 100**

---

## **Additional Clarifications**

- **Bandwidth Units:** All bandwidth values should be in Kbps for consistency.

- **Abusive Usage Threshold:**
  - Users are considered abusive if their requested bandwidth exceeds their MIR by more than **20%** (i.e., $ \text{Requested}_{i,t} > \text{MIR}_{i,t} \times 1.2 $) over at least **3 consecutive time steps**.

- **Implementation Notes:**
  - Ensure that your code correctly handles edge cases, such as users requesting zero bandwidth.
  - Test your reward function independently to verify correctness before integrating it into the RL agent.

- **Creativity and Innovation:**
  - You are encouraged to propose alternative methods or enhancements, provided they meet the challenge requirements.

- **Academic Integrity:**
  - Ensure all parts of your submission are original or properly cite any external sources.

---

## **Getting Started**

1. **Analyze the Dataset:**
   - Understand user behavior, peak usage times, and patterns.
   - Identify potential challenges in meeting demands.

2. **Set Up the Environment:**
   - Implement the initial allocation strategy precisely.
   - Ensure your environment simulates the dynamics accurately.
   - Focus on the RL agent's role in optimizing the remaining bandwidth.

3. **Design the RL Agent:**
   - Define state and action spaces effectively to reflect the focus on remaining bandwidth.
   - Choose an RL algorithm appropriate for the problem complexity.
   - Consider models capable of handling constraints on action spaces (e.g., constrained RL).

4. **Implement Reward Function:**
   - Implement the reward and penalty components accurately.
   - Ensure proper handling of the parameters $ \beta, \gamma, \theta, \Delta t_{\text{min}} $.

5. **Train and Test:**
   - Use the training data to train your agent.
   - Evaluate performance on the testing data.
   - Monitor metrics such as allocation ratio and penalties over time.

6. **Iterate and Improve:**
   - Refine your model based on evaluation results.
   - Experiment with different hyperparameters and architectures.

---

## **Conclusion**

This challenge provides an opportunity to apply reinforcement learning to a realistic telecommunications problem. By dynamically optimizing the allocation of the **remaining bandwidth after initial allocation**, you play a crucial role in enhancing user satisfaction and system efficiency.

We encourage clarity, correctness, and creativity in your approach. Remember to:

- **Guarantee a minimum of 1 Mbps** for every user at all times (handled in Phase 1).
- **Optimize the use of remaining bandwidth** through the RL agent.
- **Prevent over-allocation** of total bandwidth.
- **Effectively manage sustained abusive usage**.

We look forward to reviewing your innovative solutions. Good luck!

---

### **Final Notes**

- **Key Focus:** The RL agent operates exclusively on the remaining bandwidth after the CIR allocations, aiming to maximize the average allocation ratio without violating constraints.

- **Importance of Constraints:** Adherence to constraints is crucial for system stability and fairness. Ensure your model respects all specified limits.

- **Testing and Validation:** Thoroughly test your model to ensure it performs well under various demand scenarios and effectively balances user satisfaction with constraint adherence.

---

Thank you for participating in this challenge. We are excited to see your solutions and how you address this real-world problem using reinforcement learning!