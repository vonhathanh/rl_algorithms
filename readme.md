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

