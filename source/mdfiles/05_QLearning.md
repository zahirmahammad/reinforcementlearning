# Q Learning

An Off-policy, value-based method that uses TD approach to train its action-value function

Q-Learning = An algorithm to train Q-function - an action-value function

-- diagram ---

Q = Quality (the value) of that action at that state

- Internally, Q-function is encoded by Q-table, a table where each cell corresponds to a state-action pair

    $\pi^*(s) = argmax_aQ^*(s, a)$ --- {epsilon-greedy policy}

-- diagram --

- Optimal Q-table = Optimal Q-function = Optimal Policy

## Algorithm

1. Initiate Q-table
2. Choose an action based on $\epsilon$-greedy strategy {$\epsilon$ = explore; 1-$\epsilon$ = exploit}
3. Perform $A_t$, get $R_{t+1}$ & $S_{t+1}$
4. Update $Q(S_t, A_t)$ {TD learning}

    $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1}) - V(S_t)]$
    
    $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, a) - Q(S_t, A_t)]$

The TD target = 
$R_{t+1} + \gamma Q(S_{t+1}, a)$

$R_{t+1}$ = Current Reward

$Q(S_{t+1}, a)$ = Greedy policy to choose the highest Q-valued action of next state {not $\epsilon$-greedy policy}

- For updating we use $\rightarrow$ Greedy policy

- For New state $\rightarrow$ $\epsilon$-greedy policy

That's why "Off-policy"

## Off-Policy Vs On-Policy
- Off-Policy - uses different policy for acting(inference) and updating(training)

    - In Q-learning $\rightarrow$ $\epsilon$-greedy {acting policy} ; greedy policy {updating policy}

- On-Policy - uses same policy for acting and updating

    - Eg - Sarsa, another value-based algo. uses $\epsilon$-greedy to select next state-action pair

    $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$


