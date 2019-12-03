# Project: performance of pre-training methods

## Research question

How does the performance compare to baseline for a deep RL algorithm with

1. q network inputting pretrained decoded state
2. q network inputting next state (via pretrained transition model) in addition to state and action
3. both 1 and 2

## Components

- Environment
    - Continuous vs. discrete
    - Dimensionality of observation and action space
    - Complexity of transition dynamics
    - Sparsity of reward and difficulty of credit assignment
    - Room for improvement: the baseline performs well, but is by not means "saturated" in performance
        - For example, I think Pendulum is too easy, though it would be a good sanity checker
    - I will at least start this with continuous environments, since I haven't implemented algorithms for discrete environments
- Algorithm
    - Not too fancy but works reliably (as RL algorithms go)
    - Mileage may vary for pretraining approach, so should try different algorithms
    - TD3 seems reasonable as a continuous algorithm
- Model
    - Should be fine to keep same basic architectures, but different size to fit encoded representations
    - State autoencoder:
        - Map: o -> E(o) -> D(E(o))
        - Loss: RMSE( o, D(E(o)) )
    - Transition dynamics estimator
        - Map: (o, a) -> D(o, a)
        - Loss: RMSE( o', D(o, a) )

## Uncertainties

- What is the rationale for a transition dynamics estimator helping a q estimator?
    - Transition dynamics estimator predicts the next state
    - Q is the expected return for taking an action from a state
    - So TDE helps Q by explicitly representing one extra step in the long-horizon estimate it has to make
    - You can in fact simulate indefinitely with the TDE: predict next observation, feed that into policy, get next observation, feed that into policy, ... This enables simulation in MCTS, for example.
        - If this is the case, what about a recurrent architecture?
        - You could try to predict $n$ steps ahead
        - Or take each observation from 1..$n$ steps ahead, concatenate, and input to rest of Q estimator
        - LSTM?
        - **Consider this a possible extension**
- Should the transition dynamics estimator be like an autoencoder in that it squeezes in the middle?
    - What would this achieve?
    - I think it would only hinder learning to have this

## High-level plan

- Configure initial testbed
    - Environment: 1. `Pendulum-v0` 2. `LunarLanderContinuous-v2`
    - Algorithm: TD3
- State autoencoder
    - Design
    - Implement model
    - Write training procedure
    - Debug
    - Train in isolation on its task
    - Review performance and make adjustments
- State autoencoder and policy integration
    - Design policy network for compatibility
    - Implement model
    - Implement parameter transfer procedure
    - Debug
- Evaluation
    - Train augmented policy network on standard task
    - Review performance and make adjustments
    - Design full experiment
    - Compile and analyse results of full experiment

## Log

### 2019.12.03

State autoencoder architecture

- We may have to work with pixels now
    - I mean we don't have to, but I'm more confident a pretrained autoencoder will help significantly if the raw state is very high-dimensional
    - So for reference, try this to get pixel state from `gym`: `im_frame = env.render(mode='rgb_array')`
- I'm looking at Deep TAMER (Warnell et al. 2018) for inspiration since that's the first paper where I came across this technique

Thinking about experimental design

- I felt that comparison of policy aided by state autoencoder and q aided by transition model was not close enough
- Besides, Deep TAMER uses an autoencoder to help the reward model, not the policy.
- So I think the two variants (at least as the top priority) should be state encoder for q vs. transition model for q. So if the model is M, we have Q(M(s), a) vs. Q(M(s, a))
    - But being preemptive here, I already suspect Q(M(s, a)) performs worse. We might still want the original information of (s, a). So how about Q(s, a, M(s, a)), perhaps with M(s, a) getting plugged in later in the forward pass?
    - Then, the combined approach is Q(M1(s), a, M2(s, a))
- 
