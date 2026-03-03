# Monte Carlo Reinforce (Policy-Gradient)

- We have a parametrized stochastic policy
- We have a probability distribution over action. The probability of taking each action is also called action preference.

The idea
- We let the agent interact during an episode. If we win the episode, we consider each action taking was good and must be more sampled.

Algorithm

- Training loop
  - Collect episode with $\pi$
  - Calculate return for each action
  - Update weights to $\pi$
    - if positive return $\rightarrow$ increase every $P(S, a)$ pair taken in that episode
    - if negative return $\rightarrow$ decrease every $P(S, a)$ pair taken in that episode
