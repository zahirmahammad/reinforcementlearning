# Actor-Critic Methods

- Policy-Based methods - We aim to optimize policy directly without using a value function
- Reinforce - Policy-Gradient methods - optimizes policy directly by estimating weights using Gradient Ascent
- Policy Gradient estimation is the direction of steepest increase in "return"
- In reinforce, since we use MC sampling to estimate return (entire episode to calculate return), we have significatn varience in policy gradient estimation
- Monte Carlo leads to slower training since we need a lot of samples to mitigate it.

## Actor-Critic
A hybrid architecture combining "value-based" and "Policy-based" ti stabilize the training by reducing the variance using 
- An Actor - control how our agent behaves (Policy-based)
- A Critic -  measure how good the taken action is (value-based)

We'll train Advantage Actor Critic (A2C)

## The Problem of Variance in Reinforce
- In Reinforce, we want to increase the probability of actions in a trajectory proportionally to how high the return is

$$\nabla_\theta J(\theta) = \sum_{t=0} \nabla_\theta log \pi_\theta (a_t | S_t) R(\tau)$$

- The return $R(\tau)$ is calculated using a MC sampling. We collect a trajectory and calculate the discounted return, and use this score to increase or decrease the probability of every action taken in that trajectory
- The advantage is "its unbiased". Since, we are not estimating any return {using true return}
- Given, environment stochasticity (random events of episode), and stochasticity of policy, trajectories can lead to different returns, which lead to high variance.
- Consequently, same starting state can lead to very different returns


One solution to mitigate high variance by using a large number of trajectories, hoping that the variance introduced in anyone trajectory will be reduced in aggregate and provide a "true" estimate of return

However, increasing batch-size, reduces sample efficiency {need another method}

## Advantage Actor-Critic (A2C)
- Reduce variance of Reinforce, train our agent faster and better
- Use a combo of Policy-Based and Value-Based Method
- The idea behind Actor-Critic. We learn 2 function approximatoins
  1. A policy that controls how our agent acts {$\pi_\theta (s)$}
  2. A value function to assist the policy update by measuring how good the action is {$\hat{q}_w (S, a)$}


## Actor-Critic Process
- At each timestep "$t$", we get current state $S_t$ from eanv and pass to Actor and Critic.
- Policy takes $S_t$ and outputs an Action $A_t$
- Critic takes the action "$A_t$" and state "S_t", computes the value of taking that $A_t$ at $S_t$ {Q-Value}
- The $A_t$ performed in environment $\rightarrow S_{t+1}$ and $R_{t+1}$
- Actor updates its policy parameters using Q values
  
    $$\Delta \theta = \alpha \nabla_\theta(log \pi_\theta(S, a)) \hat{q_w} (S, a)$$

  - $\Delta \theta \implies$ change in policy parameters
  - $\hat{q_w} (S, a) \implies$ action-value estimation
- With updated parameters, Actor produces next action $A_{t+1}$ for $S_{t+1}$
- The Critic then updates parameters
    
    $$\Delta w = \beta[R(S, a) + \gamma \hat{q}_w (S_{t+1}, a{t+1}) = \hat{q}_w (S_t, a_t)] \nabla_w \hat{q} (S_t, a_t)$$

  - $\beta \implies$ learning rate
  - $\nabla_w \hat{q} (S_t, a_t) \implies$ grad of value function


## Adding Advantage in Actor-Critic (A2C)
- More stabilized learning by Advantage function as "critic" instead of action-value function
- The idea is Advantage function calculates relative advantage of action compared to others possible at a state. {how taking that action at a state is better compared to average value of state} {Subtracting mean value of state from state-action pair}

    $$A(S, a) = Q(S, a) - V(s)$$

- In other words, it calculated extra reward we get if we take action than mean reward we get at that state
- If $A(S, a) > 0$; our gradient is pushed in that direction
- If $A(S, a) < 0$; our gradient is pushed in opposite direction {as it does worse}
- We have 2 value functions here $\rightarrow Q(S, a)$ and $V(s)$
  - but, $Q(S, a) = r + \gamma V(S^`)$
  - We use TD error as a good estimate of advantage function
    - $A(S, a) = r + \gamma V(S^`) - V(S)$
- Critic Update derivation {we use $\phi$ for parameters}

    Error = $R_t - V_\phi(S)$
    
    $R_t = r + \gamma V_\phi (S_{t+1})$
    
    Error = $r + \gamma V_\phi (S_{t+1}) - V(S_t)$

    {$r + \gamma V_\phi (S_{t+1}) - V(S_t)$ is TD Error}

    In ML for loss we use Mean square error $L = \frac{1}{2} \delta_t^2$

    Applying Gradient gives

    $\nabla_\phi L = - \delta_t \nabla_\phi V_\phi (S_t)$

    update becomes

    $\phi \rightarrow \phi + \alpha \delta_t \nabla_\phi V_\phi (S_t)$


Notice, we didn't take $\nabla_\phi V_\phi (S_{t+1})$; if we take that it makes training unstable, since the goal would be moving

Actor-update

$$\nabla_\theta J = E[\nabla_\theta log \pi_\theta (a | S). A(S, a)]$$

$$A(S, a) = \delta_t$$