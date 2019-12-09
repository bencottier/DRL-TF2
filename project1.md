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
    - Input/Output: two 160x160 frames
    - Latent: 100x1
    - Layers: 3x(64x3x3 Conv, BatchNorm, 2x2 MaxPool), 1x(1x3x3 Conv, BatchNorm, 2x2 MaxPool)
    - Let's start by copying this

Thinking about experimental design

- I felt that comparison of policy aided by state autoencoder and q aided by transition model was not close enough
- Besides, Deep TAMER uses an autoencoder to help the reward model, not the policy.
- So I think the two variants (at least as the top priority) should be state encoder for q vs. transition model for q. So if the model is M, we have Q(M(s), a) vs. Q(M(s, a))
    - But being preemptive here, I already suspect Q(M(s, a)) performs worse. We might still want the original information of (s, a). So how about Q(s, a, M(s, a)), perhaps with M(s, a) getting plugged in later in the forward pass?
    - Then, the combined approach is Q(M1(s), a, M2(s, a))

### 2019.12.04

Writing autoencoder class

- Bin for applying dropout to initial identically sized layers:

    ```python
    up_stack = list()
    prev_filters = None
    i = -2
    while i >= -len(self.hidden_sizes) and self.hidden_sizes[i] != prev_filters:
        up_stack.append(self.upsample(self.hidden_sizes[i], self.kernel_size, apply_dropout=True))
        prev_filters = self.hidden_sizes[i]
        i -= 1
    up_stack += [self.upsample(f, self.kernel_size) for f in self.hidden_sizes[i::-1]]
    ```

- Got it built and returning an output
- Have not tested the encoder output, flattened and isolated
- Next we will write a training loop to test its functionality in its own right

### 2019.12.05

Testing encoder

- Seems to have a reasonable output: (1, 100), float32, values are small with mean about 0

Autoencoder training

- Hmm...which is better? Randomly sampling states, or random sampling actions and collecting states from the resulting trajectories?
    - The latter seems like it would be closer to actual behaviour, and in turn more relevant to the final model
- Don't think this laptop can handle 1M buffer size of 64x64x3 images of 4-byte numbers
    - I have 7.5 GiB of RAM
    - Let's say budget is 4 GiB
    - 4*1024^3 / (64x64x3x4) = 87K
    - So 100K seems safe
    - Not sure how much we would sacrifice by converting to grayscale. By eye, it is harder to distinguish the moving object.
    - Actually, we can probably adopt a different strategy. Since we are doing supervised learning on a bunch of states, it seems appropriate to separate data collection and training.
- It's multiple frames of 3 channels...what's the deal with shape and batch size?
    - DDPG paper

        > the observation reported to the agent contains 9 feature maps (the RGB of each of the 3 renderings) which allows the agent to infer velocities using the differences between frames. The frames were downsampled to 64x64 pixels and the 8-bit RGB values were converted to floating point scaled to [0, 1]. 

- Deep TAMER paper

    > We acquire the training states for each environment in offline simulation using a random policy

- For this supervised learning task I think 100K-sized dataset is plenty to get a good result. CIFAR and MNIST are 60K for example. The MuJoCo, even Atari state data distribution is not that wide - much narrower than CIFAR. The main difficulty appears to be getting the random policy to generate states likely to be encountered throughout (some portion of) the training proper.

### 2019.12.06

Creating state dataset

- https://www.tensorflow.org/tutorials/load_data/images
    - `.tgz` file
    - 5 subdirectories (one per class in the example)
- I think we will be fine with a single directory of JPEGs

### 2019.12.07

Running state dataset generation: `State dataset of 1e+05 samples for Hopper-v2 saved to ./data/state/Hopper-v2`

- Expect to take about 5 hours
- Images are 128x128; I thought 64x64 looked too corrupted (thought it would probably be fine given success of DDPG paper experiments), and it doesn't take much longer to generate

Writing batch loop for state autoencoder

- First we need to initialise the dataset that will exist shortly
    - There will be a directory of `.jpg` files
    - This will be compressed into a single `.tgz`
    - Base it on this

        ```python
        import pathlib
        data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
        data_dir = pathlib.Path(data_dir)
        ```

    - Actually no, the above just ends up with a directory of `.jpg` files anyway
- Based dataset load on https://www.tensorflow.org/tutorials/load_data/images
- Model can train without error now, but memory is overloading to the point of frozen desktop. It might be the caching getting out of hand.

### 2019.12.09

Investigating training memory problem

- Simplest thing is to disable caching and see what happens
- 
