# DQN

Deep Q-Learning

When state space is gigantic, creating, maintaining a Q-Table is not efficient. THe best idea is to approximate the Q-values using a parametrized Q-function $Q_\theta (S, a)$

--- diagram ---

- Atari games have image input
- As input, we stack 4 frames, pass through network, and output Q-values

## Pre-processing the input and temporal limitation

We need to preprocess the input $\rightarrow$ to reduce complexity of state $\rightarrow$ reduce computation time for training

- In images, if color don't add much information, then reduce the complexity we can change to grayscale and reduce size

- Why do we stack frames:

    - It helps handle temporal difference limitation

    - Example: When you take ping-pong game (2D) you can't tell, which direction the ball is going, but when you stack, 3 or more frames, you can tell.

    - So, to capture temporal informateion, we stack frames.

- The stacked frames are then passed to CNN's. These layers allow us to exploit and capture spatial relationships in images

- Finally, we have fully connected layers that output a Q-value

## The Deep Q-learning Algorithm

- In DQL, we create a loss function that compares our "Q-value prediction" with "Q-target" and uses gradient descent to update the weights of our DQN to approximate Q-values better

    - Q-target = $y_j = r_j + \gamma max_a Q(S_{j+1}, a^` ; \theta)$

    - Q-loss = $y_j - Q(S_j, a_j ; \theta)$

- DQL has 2 phases

    - Sampling - We perform actions and store experience tuples in replay memory
    - Training - Select a small batch of tuples randomly and learn from this batch using gradient descent update
  
- DQL training might suffer from instability, mainly because of combining a non-linear Q-value function(nueral net) and bootstrapping (when we update targets with existinf estimated not actual return)
  
- To stabilize, we implement 3 solutions
    1. Experience replay to make more efficient use of experience
    2. Fixed Q-target to stabilize the training
    3. Double DQL, to handle the problem of overestimation of Q-values

## 1. Why replay memory?
- Usually in online RL, the agent interacts with the env, gets experiences, learns from them and discards and is not efficient.

    Experience replay helps by using the experiences of training more efficiently. We use replay buffer that saves experience samples

- Avoid forgetting previous experiences and reduce correlation between experiences (catastrophic forgetting)

    We initialize a replay memory buffer 'D' with capacity N.

## 2. Fixed Q-Target

- In the TD error, we calculate difference between TD-target (Q-target) and current Q-value (estimation of Q)

    But we don't know real TD target. To avoid oscillations while training in the initial it is just garbage values. We use 2 seperate networks and update (copy) the parameters from DQN to target network every "c" steps

## 3. Double DQN

- The problem of overestimation of values
  
    $$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1}) - V(S_t)]$$

- Simple Ques: How are we sure about best action for next state has the highest Q-value in "Neural Net". Since the accuracy of Q-values depends on what action we tried and states we explored.
- In the initial since Q-values are noisy, the best action can lead to false positives. If non-optimal are give priority, then there is no point.

For solution, we use
1. Q-Network: to select best action for next state (highest Q-Value)
2. Target Network: to calculate the target Q-value of that action at next state (estimation)

Normal DQN  $ \implies y = r + \gamma Q_{target}(S, a)$

Double DQN $\implies y = r + \gamma Q_{target}(S, argmax_a Q_{online}(S,a))$