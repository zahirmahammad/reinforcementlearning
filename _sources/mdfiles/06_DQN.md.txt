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

    - Sampling
    - Training