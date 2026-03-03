# Value-Based Methods

Deep Q-Learning

First Deep RL Algorithm that played Atari games and beat the human level

Two types of Value-based methods

In value-based, we learn a "value function" that maps a state to a expected value of being at that state (expected return)

"Value of state" = Expected Discounted Return, that agent gets if it "starts and acts according to policy"

But we don't have a policy in value-based, we train value-function not policy

Policy-Based

1. Deterministic - Returns one action given a state

2. Stochastic - Returns a probability distributions over actions

Value-Based

Indirectly, by training a value function, Given this value function, our policy will take an action

- Policy is not trained/learned, we need to specify its behavior
- Greedy Policy - Given, the value function, will take actions that always leads to biggest rewards

In value-based
- Don't train a policy
- Our policy function is defined by hand
- Instead train a value function that is a Neural Network

In every method, there is a policy, but in value-based you don't train policy, {but it's a pre-specified function}

$\pi^*(s) = argmax_a Q^*(s, a)$

Infact, most value-based use "Epsilon-Greedy" Policy

Two types of value-based functions

1. State-Value function
    $V_\pi (s) = E_\pi[G_t | S_t = s]$

    For each state, the state-value function outputs the expected return if the agents starts at that state

2. Action-Value function

    In action-value function, for each state-action pair, the action-value function outputs the expected return if agent starts in that state, takes that action, & follows policy ever after

    $Q_\pi(s, a) = E_\pi[G_t | S_t=s, A_t=a]$

Difference
- For state-value function, we calculate the value of state $S_t$
- For action-value function, we calculate, the value of state-action pair $(S_t, A_t)$ hence the value of taking that action at that state

--- Add picture -----

## Bellmann Equation
It simplifies our state-value or action-value function

$V_\pi(s) = E_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) | S_t = s]$

The idea of Bellmann equation is that instead of calculating each value as the sum of expected return, we calculate the value as sum of immediate reward + discounted value of state that follows

## Monte Carlo Vs Temporal Difference Learning

2 Learning Strategies for training value function or policy function

- Monte Carlo - uses an entire episode of experience before learning
- Temporal Difference - uses only a step $(S_t, A_t, R_{t+1}, S_{t+1})$ to learn

## Monte Carlo

Waits until the end of episode, calculate $G_t$(return) and uses it as a target for updating $V(s_t)$

$V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$

- Given an example of start and enf point in a grid MC waits until episode ends and gets the list of "State, Action, Reward, Next State"

[[State, $A_t$, $R_{t+1}$, $S_{t+1}$], [$S_{t+1}$, $A_{t+1}$, $R_{t+2}$, $S_{t+2}$], ......]

- Agent will sum all rewards to get "$G_t$" & updates $V(S_t)$

## Temporal Difference
TD, waits for one interaction ($S_{t+1}$) to form a TD target and update $V(S_t)$ using $R_{t+1}$ and $\gamma * V(S_{t+1})$

- Since, we don't have $G_t$, we use $R_{t+1}$ and discounted value of next state

- This is called "Bootstrapping", becuase TD bases its update in part on an existing estimate V(S_{t+1})

$V(S_t) \leftarrow V(S_t) + \alpha(R_t + \gamma V(S_{t+1}) - V(S_t))$

This method is called TD(0) or one-step TD

Simple Difference

- MC -  We update value function from complete episode {actual accurate discounted return of episode}

    $G_t = \sum_{k=0}^\infty R_{t+k+1}$

- TD - We update value function from a step, we replace "G_t" with an estimated return called "TD target"

    TD target = $R_{t+1} + \gamma V(S_{t+1})$

    Bellmann = $E_\pi[R_{t+1} + \gamma V(S_{t+1})|S_t = s]$