## Implementation detail

- We'll mimick stablebaseline API: model = DQN(env, args)
- model.train(step=1000)
- model.predict(input)

## DQN

- choose the maximum max(Q_target(S_t+1, a'), a') based on params of Q_target

![img.png](images/img.png)
## DDQN: 

- improve DQN by decouple the action selection process from it's evalution, 
- We use argmax(Q_value(S_t+1, a'), a') to choose the action, not Q_target
- Reduce harmful overestimations by the max() operator
![img_1.png](images/img_1.png)

## Prioritized replay

- Sample transitions from which there is much to learn
- Probability of transition pt is based on the diff between max(q_target(s_t+1, a')) - q_value(s_t, a)
- We then raise pt by omega w, w is a hyperparameter that control the shape of the distribution
![img_2.png](images/img2.png)


## Noisy Net
- most exploration heuristics rely on random perturbations of the agent's policy (epsilon greedy)
- unlikely to lead to the large-scale behavioural patterns needed for efficient exploration in many environments
- Optimism in the face of uncertainty is a common heuristic in RL that has better performance but not easily applied
with more complicated function approximators
- augment the environment’s reward signal with an additional intrinsic motivation term that reward novel discoveries:
    - separate the mechanism of generalisation from exploration
    - weighting relative to reward must be chosen by experimentals, may introduce bias to the policy
- NoisyNet: learned pertubations of the network weights are used to drive exploration
- The key insight is that a single change to the weight vector can induce a consistent, 
and potentially very complex, state-dependent change in policy over multiple time steps
- noise is added to the policy at every step, pertubations are sampled from a noise distribution
- variance of the pertubation is a parameter that can be considered as the energy of the injected noise
- double the parameters in the linear layers. weights are simple affine transform of the noise.
- computation complexity still dominated by mat mul, not weights generation
- noisy net: phi = mu + sigma * epsilon (element-wise mul), mu and sigma are learnable params, epsilon is a 
zero-mean vector of noise with fixed statistics
- we replace the normal linear layer wx+b by:
![alt text](images/image.png)
- all of this operations are just affine transformation (element-wise mul and addition)
- they use factorised Gaussian: p unit of Gaussian vars epsilon_i for input noises and q unit Gaussian noise
for outputs (total = p + q vars)
![alt text](images/image-1.png)
- fc layers of value net are parameterised as a noisy net, parameters are drawn from the noisy net param distribution
![alt text](images/image-2.png)
- note the different noise used in two networks: avoid bias

## PPO

- Optimize multiple steps of policy gradient loss using the same trajectory is not well-justified, empricially
leads to destructive large policy updates
- TRPO: maximize a surrogate objective function subject to a constraint of the size of policy update
![img.png](images/img_3.png)
- We can rewrite TRPO to solving the unconstrained optimization problems
![img_1.png](images/img_4.png)
- Certain surrogate objective forms a lower bound on the performance of policy
- Choose beta is hard, experiments showed
- PPO: modify the objective, penalize changes to the policy that move r(theta) away from 1
![img_2.png](images/img_5.png)

## TD3

- Overestimation bias and the accumulation of error in temporal difference methods are present in an actor-critic setting
- This inaccuracy is further exaggerated by the nature of temporal difference learning
- Estimate of the value function is updated using the estimate of a subsequent state
- Using an imprecise estimate within each update will lead to an accumulation of error.
- DDQN is ineffective against actor-critic setting due to slow-changing policy -> current and target
value estimates remain too similar
- Use Double Q-learning to deal with this problem: using a pair of independently trained critics
- Unbiased estimate with high variance can still lead to future overestimations in local regions of state space
- Propose Clipped Double Q-learning: value estimate suffering from overestimation bias can be used as an 
approximate upper-bound to the true value estimate -> favors underestimations, do not tend to be propagated during learning,
as actions with low value estimates are avoided by the policy

### Overestimation Bias in Actor-Critic

- Policy is updated with respect to the value estimates of an approximate critic
- overestimation may be minimal with each update, the presence of error raises two concerns:
  - may develop into a more significant bias over many updates if left unchecked
  - inaccurate value estimate may lead to poor policy updates
- Double Q learning: greedy update is disentangled from the value function by maintaining two separate value estimates
- Use a pair of actors (pi1, pi2) and critics (q1, q2)

## Target Networks and Delayed Policy Updates

- Stable target reduces the growth of error, provide a stable objective in the learning procedure
- Without a fixed target, each update may leave residual error which will begin to accumulate
- Fixed policy + slow updating target network -> stable learning process of value network
![img.png](img.png)
- Divergence that occurs without target networks is the result of policy updates with a high variance value estimate
- If target networks can be used to reduce the error over multiple updates, and policy updates on high-error states cause
divergent behavior, then the policy network should be updated at a lower frequency than the value network
-> Delaying policy updates until the value error is as small as possible

## Target Policy Smoothing Regularization

- Deterministic policies can overfit to narrow peaks in the value estimate
- Update the critic with a learning target using a deterministic policy increase the variance of the target
-> We need regularization
- Enforce similar actions should have similar value
- Fitting the value of a small area around the target action
-> Adding a small noise