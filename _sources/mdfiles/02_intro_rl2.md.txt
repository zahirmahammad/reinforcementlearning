# How do we Solve RL Problems

Two main approaches to solve

1. Value-Based
    - Indirectly teach the agent to learn "which state is more valuable" and then take the action that leads to that state
2. Policy-Based 
    - Directly, by teaching the agent to learn which action to take given current state


## 1. Value-Based
- We learn a value function that maps a state to expected value of being at that state

- Value of State = expected discounted return the agent can get if it starts in that state and then acts according to the policy

- According to policy = going to state with highest value

    $V_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...... | S_t = S]$

- Value function basically "give value to each state"

    {Think of a 4x4 grid and each checkbox has a value}

- With this value function at each step, our policy will select the state with highest value

## 2. Policy-Based

- Learn the policy function directly $\pi^*$

- Function defines a mapping from "state" to "best action to that". Alternatively, it defines a "probability distributions" over set of possible actions to that state.

- Two types of policies

    1. Deterministic - Policy always returns the same action

        $a=\pi(s)$

    2. Stochastic - Outputs a probability distributions over actions

        $\pi(a|s) = P[A|S]$



The Deep in RL - is nothing but usage of Deep NN

Value-Based Algos

1. Q-Learning - QTable
2. Deep Q-Learning - A NN to predict