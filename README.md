# Reinforcement Learning Repo Initialization

This repository is where I will explore different RL strategies.

---

# Markov Decision Process (MDP)

**Flow:**  
state → action → reward → state → ...

- Each state depends only on the previous state.

---

## Key Concepts

- **Policy**: A function that takes in a state and returns an action.  
- **Reward (r)**: The immediate signal received after taking an action.  
- **Discount Factor (γ)**:
    - If **gamma = 0**, the agent only considers immediate rewards
    - If **gamma = 1**, the agent treats all future and immediate rewards equally  
    - If **0 < gamma < 1**, it balances short-term and long-term rewards

---

## Return (Total Reward)

Denoted as **G<sub>t</sub>**:

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
\]

## Goal

Find the policy that maximizes total reward