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

## Diving Deeper

Stochastic Policy $\implies \pi_\theta (S) = P[A | S ;\theta]$

Objective function $\implies J(\theta)$

The objective function gives us the performance of the agent given a trajectory (state-action sequence without considering reward) and outputs the expected cummulative reward

$J(\theta) = E_{\tau \sim \pi} [R(\tau)]$

$R_{(\tau)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3}+....$

$\tau$ = Trajectory (Sequence of states and actions)

- Since $J(\theta)$ is "expected" return, it is the weighted average (where the weights are given by $P(\tau, \theta)$) of all possible values that $R(\tau)$ can take

    $$J(\theta) = \sum_\tau P(\tau; \theta) R(\tau)$$

    $\sum_\tau$ = we calcuate $J(\theta)$ by summing all trajectories, the probability of taking that trjectories give $\theta$

    $P(\tau; \theta)$ = Probability of trjectories (depends on $\theta$; which is policy parameters)

    $$P(\tau; \theta) = \Pi_{t=0}P(S_{t+1} | S_t, a_t) \pi_\theta(a_t | S_t)$$

    $P(S_{t+1} | S_t, a_t)$ = Environment dynamics (state distributions)

    $\pi_\theta (a_t | S_t)$ = Probability of taking $a_t$ at $S_t$

- Our objective then is to maximize the expected cummulative reward by finding '$\theta$' that outputs best action probability distributions

    $$max_\theta J(\theta) = E_{\tau \sim \pi_\theta} [R_\tau]$$


## Gradient Ascent and Policy-Gradient Theorem

- Policy-gradient is an optimization problem.
- Find values of $\theta$ that maximized our objective function $J(\theta)$ 
- Solution: Gradient Ascent (opp. if Gradient Descent)
  - Update step $\implies$ $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$


- We still have 2 problems with computing $\nabla_\theta J(\theta)$
  1. We can't calculate true gradient of objective function since it requires calculating probability of each possible trajectory. So, we calculate a gradient estimation with a smaple-based estimation (collect some trajectories)
  2. To differentiate objective function, we need to differentiate state dynamics, which we may not know.

Solution: **Policy Gradient Theorem**

For any differentiable policy and for any policy objective function, the policy gradient is 

$\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log \pi_\theta (a_t|S_t) R(\tau)] = \hat{g}$
