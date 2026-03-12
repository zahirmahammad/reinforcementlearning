# Reinforcement Learning

**Definition**

Reinforcement Learning is kind of machine learning but without any datasets. It's learning based on try, fail through experience.

We build an agent that makes smart decision by interacting with the environment

Agent decision-making is called the policy 
"$\pi$"

Technically it is said that - "RL is based on 'Reward Hypothesis'"

## Basic Terminologies

Here are the basic terms before diving into RL, later we'll discuss about the other terms that people doing RL use

- Agent
- Markov Property
- Observation Space Vs State
- Action Space
- Rewards and Discounting
- Exploration Vs Exploitation
- Types of tasks
    - Episodic
    - Continuing


## 1. Agent
Agent is the robot or policy or the thing that wants to learn something.

The agent needs to acheive a specific goal, we define the problem as a maximization problem. The agent needs to get the maximum reward that is possible in the episode.

## 2. Markov Property
It says that our agent needs only the current state to decide what action to take and not the history of states and actions that took before.

## 3. Observation Space Vs State Space
- State - Complete description of state of the world (chess game {you have access to the entire world})

- Obs - A partial description of state (Super mario game)

## 4. Action Space
- Discrete Action Space: finite number of actions
- Continuous Action Space: infinite number of actions

This will have importance in choosing the RL Algorithm

## 5. Rewards and Discounting
The cummulative reward can be written as 

$R_{T} = r_{t+1} + r_{t+2} + r_{t+3} + .....$

$R_{T} = \sum_{k=0}r_{t+k+1}$

sum of all rewards in future

Disounting - means "discount for future rewards"

Discount Factor $(\gamma)$ - "How much fraction of future rewards to consider"

$R_{T} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + .....$

$R_{T} = \sum_{k=0} \gamma^k r_{t+k+1}$

- If there is no "discount factor $\gamma$" all rewards would be summed and there is no focus on global goal. It just cares about what now.
- $\gamma \in [0.95, 0.99]$ not 1 (no point of being 1.0)
- larger $\gamma$ is less disount on future rewards - so cares more about long term reward
- smaller $\gamma$ is more discount on future rewards - so cares less about long term reward


## 6. Exploration Vs Exploitation

Agent need a balance between expolaration and exploitation. We use the term $\epsilon$ ; $\epsilon \in (0, 1)$

- With probability of $\epsilon$, choose explore
- With probablity of $1 - \epsilon$, choose exploit

## 7. Types of **Tasks**

1. Episodic - has START and TERMINAL state (eg: super mario)
2. Conitinuing - no TERMINAL state, continuous forever (eg: stocks, driving)