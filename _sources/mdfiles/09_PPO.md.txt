# PPO - Proximal Policy Optimization

**Key Idea:**  Modified architecture that improves our agent's training stability by avoiding policy updates that are too large

- We use a ratio that indicates the difference between our current and old policy and clip this ratio to a specific range [$1-\epsilon$, $1+\epsilon$]

- The ratio function $\rightarrow r_t(\theta)$

    $$r_t (\theta) = \frac{\pi_\theta (a_t | S_t)} {\pi_{\theta_{old}}(a_t | S_t)}$$

This will ensure that our policy update will not be too large and that the training is more stable

## The Intuition

Improve the training stability of policy - by limiting the change you make to the policy at each training epoch

## Why this Approach:
1. We know empirically - smaller policy updates during training are more likely to converge at an optimal solution
2. A too-big step in a policy update can result in falling "off the cliff" (getting a bad policy) and taking a long time or even having no probability to recover

--- off the cliff image ---


We update conservatively
- How much current policy change compared to the former one using a ratio calculation between the current and former policy and we clip this ratio in a range [$1-\epsilon$, $1+\epsilon$] (we remove the incentive for the current policy to go too far from old one)

## The Objective Function
(We call it "Clipped Surrogate Objective Function")

- The policy objective function that we know

    $$L^{PG}(\theta) = E_t[log \pi_\theta (a_t | S_t) . A_t]$$

  - $log \pi_\theta (a_t | S_t) \rightarrow$ log probabilty of taking that action at that state
  - $A_t \rightarrow$ Advantage if $A>0$, this action is better than the other action possible at that state

- We would push our agent to take action that lead to higher rewards and avoid harmful actions
- However, the problem comes from step size
  - too small, training process goes too slow
  - too high, there will be too much of variability in training
- We constrain our policy update with a objective function called "Clipped Surrogate Objective Function" - that will constrain the policy change in a small range using a clip
- This new objective function in PPO - to avoid destructively large weights update

    $$L^{clip}(\theta) = \hat{E}_t[min(r_t (\theta) \hat{A}_t, clip(r_t (\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$

- The ratio function $\rightarrow r_t(\theta)$

    $$r_t (\theta) = \frac{\pi_\theta (a_t | S_t)} {\pi_{\theta_{old}}(a_t | S_t)}$$

  the ratio of probability of taking action $a_t$ at $S_t$ in current policy  to same for previous policy

  - if $r_t(\theta) > 1$, the action $a_t$ at state $S_t$ is more likely in the current policy than the old policy
  - If $r_t (\theta)$ is between 0 and 1, the action is less likely for current policy than for the old one
  
  This probabilty ratio is an easy way to estimate the divergence between old and current policy

## The unclipped part of Surrogate objective function

$r_t (\theta) \hat{A_t} \rightarrow$  the ratio can replace the log probability we use in policy objective function

- While implementing, to calculate ratio, we use 

  $z = log(r_t(\theta)) = log(\frac{\pi_\theta (a | S)} {\pi_{\theta_{old}}(a | S)}) = log(\pi_\theta(a|S)) - log(\pi_{\theta_{old}}(a|S)) $

  Since we get log_prob from torch.categorical in pytorch

  $e^{z}$ gives the ratio $r_t (\theta)$ that we need

- However, if the action taken is much more probable in our current policy than in our former, this would lead to significant policy gradient step and hence an excessive policy update

## The clipped part of Surrogate objective function

$= clip(r_t (\theta), 1 - \epsilon, 1 + \epsilon)$

- We need to constrian this objective function by penalizing change that lead to a ratio far away from 1 (in paper, ratio varies from 0.8 to 1.2)
- By clipping the ratio, we ensure that we do not have a too large policy update because current can't be too different

## 2 Solutions

1. TRPO (Trust Region Policy Optimization) - uses KL divergence constraints outside the objective function to constrain the policy update.
   - But this method is complicated to implement and takes more computation time
2. PPO clip probability ratio directly in the objective function with its clipped surrogate objective function


- The clipped part is a version where $r_t (\theta)$ is clipped between [$1-\epsilon, 1+\epsilon$]
- $\epsilon$ is a hyperparameter that helps us to define this clip range (in paper $\epsilon = 0.2$)
- Then, we take min. of clipped and non-clipped objective. So, the final objective is a lower-bound of unclipped objective

## Visualize the Clipped Surrogate Function

-- Add the graphs --

We have 6 cases
1. $r_t (\theta) \in [1-\epsilon, 1+\epsilon]$ ; $A_t$ is +ve ; $r_t (\theta) A_t$
2. $r_t (\theta) \in [1-\epsilon, 1+\epsilon]$ ; $A_t$ is -ve ; $r_t (\theta) A_t$
3. $r_t (\theta) < [1-\epsilon, 1+\epsilon]$ ; $A_t$ is +ve ; $r_t (\theta) A_t$
4. $r_t (\theta) < [1-\epsilon, 1+\epsilon]$ ; $A_t$ is -ve ; $(1 - \epsilon) A_t$
5. $r_t (\theta) > [1-\epsilon, 1+\epsilon]$ ; $A_t$ is +ve ; $(1 + \epsilon) A_t$
6. $r_t (\theta) > [1-\epsilon, 1+\epsilon]$ ; $A_t$ is -ve ; $r_t (\theta) A_t$

- In 1 and 2; clipping doesnot apply since the ratio is between [$1-\epsilon, 1+\epsilon$]
- In 3, Advantage is positive, So you want to increase the probability of taking that action at that state
- But in 4, advantage is negative, the $r_t$ is already less than range, we don't want to decrease further, Hence gradient becomes zero
- In 5, advantage is positive, we don't want to get too greedy we already have higher probability of taking that action in previous policy. Hence, gradient is "0"
- In 6, the advantage is negative, we want to decrease the probability of taking that action at that state.

- So, we only update the policy with unclipped objective part, when min. is the clipped objective part, we don't update since the gradient will be "zero"
- We update only if
  - Our ratio $\in [1-\epsilon, 1+\epsilon]$
  - Our ratio is outside range, but advantage leads to getting close to the range
    - being below but, advantage > 0
    - being above but, advantage < 0

The final clipped surrogate objective loss for PPO Actor-critic style looks like - a combination of clipped surrogate objective function, value loss function and entropy bonus

$$L_t^{clip+VF+H} = \hat{E_t} [L_t^{clip} (\theta) - c_1 L_t^{VF(\theta)} + c_2 H[\pi_\theta](S_t)]$$

- $c_1, c_2$ are coefficients
- $L_t^{VF(\theta)}$ is squared-error value loss $(V_\theta(S_t) - V^{target})^2$
- $H[\pi_\theta](S_t)$, entropy of policy at state $S_t$

We add entropy bonus for sufficient exploration

- In the above equation, we maximize the reward [objective function] later, while implementing we include negative for gradient descent
- We want $L^{clip} (\theta)$ higher and minimize $L_t^{VF}$ {squared error}, so we subtract.
- We add entropy term to improve exploration (Let's talk about this)

## Entropy
- Entropy is a measure of how stochastic (random) the policy's action choices are
- We use $H(\pi)$ for entropy
- Without entropy term
  - policy quickly becomes deterministic
  - stop exploring
  - gets stuck in local optimum
- With entropy term, It encourages
  - trying different actions (we'll talk how exactly this happens)
  - better exploration during training

$H[\pi_\theta](S) = - \sum_a \pi(a | S) log\pi(a | S)$  
- High entropy value $\rightarrow$ very random actions
  - Policy output $\rightarrow$ [0.25, 0.25, 0.25, 0.25] $\rightarrow$ high entropy
- Low entropy value $\rightarrow$ very deterministic actions
  - Policy output $\rightarrow$ [0.99, 0.01, 0.0, 0.0] $\rightarrow$ low entropy

The entropy term is added to the objective to 
- prevent policy from becoming deterministic too early
- encourage exploration during training
- $c_2$, is the entropy coefficient (hyperparameter you choose)

Facts
- $log\pi(a | S)$ is always $< 0$, as $0 < \pi (a | S) \le 1$
- So $\pi(a | S) log\pi(a | S) \le 0$, hence, $H[\pi](S) \ge 0$, entropy term is always positive
- Entropy addition to the equation tpically makes the probability distribution of policy to spread out and not get a peak eaerly
- But entropy is just added to objective function which means it is increasing the current objective value. How exactly is entropy contributing to reduce higher probabilities and even out other actions?

## Entropy's Contribution to the equation
We take gradient of the objective function

$\nabla_\theta J(\theta) = \nabla_\theta L_{policy} + c_2 \nabla_\theta H$

$H(\pi_\theta) = - \sum_a \pi_\theta(a | S) log\pi_\theta(a | S)$

$\frac{\partial H}{\partial \pi_\theta(a|S )} = -(log \pi_\theta (a|S) + 1)$

1. If action probability is large, $\pi_\theta(a|s) \approx 0.99$
   - $-(log0.99 + 1) \approx -0.995$ [decreases the objective when added]
2. If action probability is small, $\pi_\theta(a|s) \approx 0.05$
   - $-(log0.05 + 1) \approx 0.3$ [increases the objective when added]

Hence, the probabilities are less peaked with this

Note: Here, we didn't consider gradient of $L_t^{VF}(\theta)$, because it is critic, it has different parameters, $\theta$ = {$\theta_{actor}$, $\theta_{critic}$}

- Grad w.r.t actor parameters, $\nabla_{\theta_{actor}} J = \nabla_{\theta_{actor}} L_{policy} + c_2 \nabla_{\theta_{actor}}H$
- Grad w.r.t critic parameters, $\nabla_{\theta_{critic}}J = -c_1\nabla_{\theta_{critic}}L_{value}$ 


