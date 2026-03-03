# Policy-Based Methods

- No learning of any intermediate value function: directly learning policy

- Recap: RL is based on Reward Hypothesis. {All goals can be described as the maximization of the expected cummulative reward}

## Value-Based, Policy-Based and Actor-Critic Methods:

- In value-based methods
  - optimal value function leads to optimal policy $\pi^*$
  - Minimize the loss between predicted and target value
  - Policy is predefined
- In policy-based methods, we directly learn to approximate $\pi^*$, without having to learn a value function
  - We parametrize the policy. For instance, using $\pi_\theta$, this policy will output probability distributions over actions (stochastic)
  
    Stochastic Policy $\implies \pi_\theta(S) = P[A|S ; \theta]$

  - Goal to maximize the performance of parametrized policy using gradient ascent
  Actor-Critic
- Actor-Critic - It's like a combination of value-based and policy-based {more details in further sections}

Now, we can directly optimize the parameters $\pi_\theta$ that leads to best expected return. For that, we define an objective function $J(\theta)$, that is, expected cummulative reward, and we will find value $\theta$ that maximizes the function.

## The Difference between Policy-Based and Policy-Gradient
- Policy-gradient is subclass of policy-based methods
- In policy-based, the optimization is on-policy for each update, we only use data collected by our most recent version of $\pi_\theta$.

The difference between these two, lie on how we optimize parameter $\theta$
1. In policy-based methods, we search directly for $\pi^*$. We optimize '$\theta$' indirectly by maximizing the local approximation of objective function with techniques like - hill climbing, simulated annealing or evolution strategies
2. In policy-gradient methods, we optimize $\theta$ directly by performing gradient ascent on the performance of objective function $J(\theta)$


## Advantages over value-based methods
1. In policy gradient, we estimated policy directly without storing additional data.
2. Policy gradient, can learn a stochastic policy. Because of this
   - We don't implement any explore/exploit. Since we output a probability distribution of actions. It explores without always taking same trajectory
   - Get rid of problem of perceptual aliasing. It is when 2 states seem same but need different actions
3. Policy gradient are more effective in high-dimensional action spaces and continuous action spaces
4. Policy gradient has better convergence property due to stochastic behavior, because in value-based a small change in outcome of action may change the whole policy to choose different action

## Disadvantages over value-based
1. Frequently, Policy Gradient converges to local maximum instead of global optimum
2. Goes slower, so it can take longer to train