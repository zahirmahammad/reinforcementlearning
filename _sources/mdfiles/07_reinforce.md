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

## Solution: **Policy Gradient Theorem**

For any differentiable policy and for any policy objective function, the policy gradient is 

$\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log \pi_\theta (a_t|S_t) R(\tau)] = \hat{g}$

## The Reinforce Algorithm [Monte-Carlo Reinforce]

Also called Monte-Carlo policy gradient; uses an estimate return from an episode to update the policy parameters $\theta$

In a loop:

- Use policy $\pi_\theta$ to collect episode $\tau$
- Use episode to estimate gradient, $\hat{g} = \nabla_\theta J(\theta)$
  -  $\nabla_\theta J_\theta = \sum_{t=0} \nabla_\theta log\pi_\theta (a_t | S_t)R(\tau)$
-  Update weights of policy : $\theta \leftarrow \theta + \alpha \hat{g}$
  
$\nabla_\theta log \pi_\theta(a_t | S_t) \rightarrow$ is the direction of steepest increase of (log) probability of selection $a_t$ from state $S_t$, tells us how we should change the weights of policy

$R(\tau)$ is scoring function

If return is high, it will push up probabilities, else it will push down the probabilities of (state, action) combinations

We can also collect multiple episode to estimate gradient

$\nabla_\theta J(\theta) \approx \hat{g} = \frac{1}{m}\sum_{i=1}^m \sum_{t=0} \nabla_\theta log \pi_\theta (a_t^{i} | S_t^{i}) R(\tau^{i})$


## [Optional] Policy-Gradient Theorem Derivation
$$J(\theta) = \sum_\tau P(\tau; \theta)R(\tau)$$

$$P(\tau; \theta) = \Pi_{t=0} P(S_{t+1} | S_t, a_t) \pi_\theta(a_t | S_t)$$

Gradient;

$$\nabla_\theta J(\theta) = \nabla_\theta [\sum_\tau P(\tau; \theta)R(\tau)]$$

Multiply RHS with $\frac{P(\tau; \theta)}{P(\tau; \theta)}$

$$
\nabla_\theta J(\theta) = \sum_\tau \frac{P(\tau; \theta)}{P(\tau; \theta)} \nabla_\theta P(\tau; \theta) R(\tau)
$$

as $R(\tau)$ is independent of $\theta$

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} R(\tau)$$

$\frac{d}{dx}log(f(x)) = \frac{1}{f(x)} \frac{d}{dx}f(x)$

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau; \theta) \nabla_\theta log(P(\tau; \theta)) R(\tau)$$

(definition of expectation)

$$\nabla_\theta J(\theta) = E_\tau[\nabla_\theta log(P(\tau, \theta)) R(\tau)]$$

Now, 

$$\nabla_\theta log(P(\tau | \theta)) = \nabla_\theta log[\Pi_{t=1}^H P(S_{t+1} | S_t, a_t). \pi_\theta (a_t | S_t)]$$

log of product is sum of logs

$$\nabla_\theta log(P(\tau | \theta)) =\nabla_\theta[\sum_{t=1} logP(S_{t+1} | S_t, a_t) + log \pi_\theta(a_t | S_t)]$$

first term independent $\theta$,

$$\nabla_\theta log(P(\tau | \theta)) =\sum_{t=1} \nabla_\theta log[\pi_\theta(a_t | S_t)]$$

Hence, 

$$\nabla_\theta J(\theta) = E_\tau [\sum_t(\nabla_\theta log(\pi_\theta(a_t | S_t))) R(\tau)]$$

Expectation can be approximated over "$m$" trajectories

$$\nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^m \sum_{t=1}^H \nabla_\theta log \pi_\theta(a_t^i | S_t^i) R(\tau^i)$$
