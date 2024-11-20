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

- 