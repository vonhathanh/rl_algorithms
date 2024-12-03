"""
Abstract:
- Joinly learn separate exploration and exploitation policies derived from the same network
- Exploitative policy concentrate on maximising the extrinsic reward (solving the task at hand)
- Exploratory ones maintain exploration without reducing to an undirected policy
- Learning exploratory policies can be thought of a set of auxiliary tasks that can help
  build a shared architecture that continues to develop even in the absence of extrinsic rewards
- Intrinsic reward that combines per-episode and life-long novelty to encourage agent to repeatedly visit all
  controllable states in the env over an epsiode
- Episode novelty encougrages agent to periodically revisit familar but not fuly explored states over
  severals episodes (not the same episode)
- Uses an episodic memory filled with all previously visited states, encoded with self-supervised objective
  to avoid uncontrollable parts of the state space
- Episodic novelty is then defined as similarity of the current state to previously stored states
- Episodic novelty rapidly adapt within an episode while life-long novelty is driven by a Random Network
  Disillation error, changes slowly, relying upon gradient descent optimisation

Reward:
- Extrinsic reward is augmented with an intrinsic reward, augmented reward at time t is:
  r_t = re_t + beta*ri_t where re = extrinsic reward, ri = intrinsic reward, beta = weight
- IR must satisfies 3 properties:
    - Rapidly discourages revisited the same state with in the same epsiode
    - Slowly discourages visits to states visited many times across episodes
    - State ignores aspects of env that are not influenced by agent's actions
- Episodic novelty module:
    - Episodic memory M
    - Embedding function f, map current state to a learned representation
    - At start, compute episodic reward r_e_t, encode curr state, save encoded state to M
    - Compare current observation with content of memory M, large diff -> large episodic intrinsic rewards
    -> Promotes the agent to visit as many different states as possible within a single episode
- Life-long novelty module:
    - Multi the exploration bonus r_e_t with a life-long curiosity factor alpha
    - Vanish over time, r_i_t = r_e_t*min(max(alpha, 1), L), L = 5
    - We could see alpha as long-term curiosity amplifier (never less than 1 and larger than 5)
- Embedding network:
    - Consider the env that has lots of variants independent of the agent's actions, such as navigating
    a busy city with many pedestrians and vehicels, agent could visit a large number of different states
    without taking any actions (because they are similar)
    - Given two consecutive states, train a Siamese network to predict the action taken by the agent to
    go from one state to the next. All the variability in the environment that is not affected
    by the action taken by the agent would not be useful to make this prediction
- Episodic memory and intrinsic reward:
    - Memory M is a dynamically-sized slot-based memory
    - At time t, M contains the controllable states of all the visited observations in curr episode
    - n(f(xt)) is the counts for the visits to the state f(xt)
The Agent:
- Using intrinsic rewards changes the MDP to POMDP (partially observed MDP) because reward = r_e + bate*r_i
- POMDP is harder to solve ->
    - The intrinsic reward is fed directly as input to the agent
    - Agent maintains an internal state representation that summaries its history of all inputs (state, action, reward)
    -> R2D2 as baseline
- The NGU intrinsic reward doesn't vanish over time -> learned policy will always be driven by it
-> Exploratory behaviour can't be turned off
- Use a universal value fn approx Q(x, a, betas) to approx the optimal value fn with a family of augmented reward
- r^betai_t = r_e_t + beta_i*r^i_t.
- We can turn-off exploratory behaviour by acting with respecto Q(x, a, 0)/
- We could think of having an architecture with only two policy: beta=0 and beta > 0
"""