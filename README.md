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
    - If **γ = 0**, the agent only considers immediate rewards
    - If **γ = 1**, the agent treats all future and immediate rewards equally  
    - If **0 < γ < 1**, it balances short-term and long-term rewards

- **Trajectory**: basically sequence of states, actions, and rewards after following policy
- **Value Functions**: keep track of average return G<sub>t</sub> given a policy in a certain state or state and action
    - state-value V<sub>policy</sub>(s)
    - action-value Q<sub>policy</sub>(s,a)
---

## Return (Total Reward)

Denoted as **G<sub>t</sub>**:

\[
G<sub>t</sub> = r<sub>t</sub> + γr<sub>t+1</sub> + γ<sup>2</sup>r<sub>t+2</sub> + ...
\]

## Goal

Find the policy that maximizes total reward