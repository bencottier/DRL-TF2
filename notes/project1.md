# Project: performance of pre-training methods

## Research question

### v1

How does the performance compare to baseline for a deep RL algorithm with

1. q network inputting pretrained decoded state?
2. q network inputting next state (via pretrained transition model) in addition to state and action?
3. both 1 and 2?

### v2

How does a low-dimensional state representation learned 

1. by an autoencoder of high-dimensional representation
2. by said autoencoder, trained further as part of actor-critic models in RL

relate to the hand-crafted state representation? How correlated are they?

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

- Note: set batch size to 4 (comparable to batch size 64 for 32x32 images)
- Simplest thing is to disable caching and see what happens
    - Goes ok
    - So I think what happens with caching is, it appends the batches to a buffer (the cache). We have told it to do so for the entire dataset, and it may be holding it in memory. (though I thought not since it is creating cache files). But huh...even so, the entire dataset is only about .5GB on disk, so I don't understand what is ultimately causing crash. Maybe because training itself is demanding so much, that extra amount pushes it over.
    - We could try using a single iterator and see if the cache still blows up
- Other problems: 
    - Runs past expected 10000 iterations per epoch
        - I think since we set the dataset to repeat forever, the iteration doesn't terminate
        - Again, single batch iterator could be the go.
    - Oh, we shouldn't having `training=False` in test because that give the encoded vector. We still want to use the decoded output for validation.
- I'll try out a single iterator style of code block
    - Nevermind, not worth it
- I think we'll be fine without caching - not worth the hassle even if we can find a way. The other optimisations like prefetch will do.

Changing research question

- For now I'm interested in exploring the similarity (or lack thereof) of a learned low-dimensional representation to the hand-crafted one.
- This requires an additional stage in the model to get the number of latent dimensions exactly right. The first option that comes to mind is a fully connected layer. We would have one either side of the latent vector.
    - But fully connected is so yesterday, right? What about average pooling?
    - Nah look, I'll stick with FC because the dimensionality is relatively low at that point, it won't slow it down much. I'm not looking for a state-of-the-art model - it's more important that it works, period. And for that, I trust FC more.

### 2019.12.10

Adding observation dimension constraint to autoencoder

- Make the specific latent dimensionality an optional parameter - if specified, add fully connected layers to convert the dimensions. This is better abstraction because we can plug in different architectures within an `if` statement or method.
- Last conv layer is (H/2^L, W/2^L, 1) where L is the number of conv layers. We flatten this to (HW/2^2L,) and pass the vector to FC with `latent_dim` units. That is then passed to another FC with HW/2^2L units, before being reshaped to (H/2^L, W/2^L, 1) and proceeding with conv, like nothing ever happend.
- The encoder output is the flattened latent vector; the decoder starts with the second FC.
- What about the forward pass condition? I think it's fine, because `Flatten` is idempotent. Let's just check that...yes.
- Have now checked the dimensionality of everything in a separate script. Should be ok.
- Updating methods proper
- Testing updates using `plot_model` for autoencoder and encoder alone
    - Original
        - Autoencoder: OK
        - Encoder: OK
    - Reduced
        - Autoencoder: OK
        - Encoder: OK (with redundant Flatten, as expected)
- Training looks OK
- **Side note**: check what the dataset shuffle buffer size means. It would be bad if it only shuffled in batches of 1000 images, since these are going to be more similar (contiguous frames).

Testing output trained on 80 images

- Mode collapse! It has memorised something resembling the initial state.
- Can we address mode collapse just with more data? Seems like no guarantee.
    - Still collapse with one run on 9000 training samples
- I think the first quick but important thing is checking how shuffling works.
- Then what else can we do? In order of information value
    - Test a larger latent dimension: the task may be too difficult with 11 dimensions!
    - Test multiple random seeds (this matters less than in RL but still matters)
    - Learn the difference between frames, or the difference between the initial and current frame. The rationale is that the variance between any two frames is relatively small, because the background is identical and the hopper rarely moves a great deal.
        - I'm mindful that Deep TAMER worked with Atari bowling, which would have similarly low variance. Did they check whether mode collapse was happening? Hopefully. But a key difference was a larger latent dimension than I'm now using.

### 2019.12.11

Understanding dataset shuffle

- https://www.tensorflow.org/guide/data
    > The Dataset.shuffle() transformation maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer.
    - So for full shuffle, we would need to set buffer size to dataset size or greater.
    - Buffer size of 1000 is the same as maximum episode length, so it is always shuffling within episodes at the least. But states in a given episode are correlated due to the dynamics: if the hopper falls over at one point, that makes it more likely it will be fallen over later on.
    - I will try a full shuffle and see how it goes, but I'm not confident this addresses the main problem (I think the size of the latent dimension and the variance between frames are more important).
- Testing full shuffle, 5 epochs
    - Still mode collapse
- Testing without `obs_dim` latent dimension (8x8 bottleneck, fully convolutional)
    - Still mode collapse
- Hot take: for the purpose of the Hopper task, mode collapse doesn't matter?
    - If all the states look similar, and the point of pretraining is just to get _closer_ to the right representation and speed up training proper, is it so bad?
    - Well, the proof is in the pudding...we would need to test the RL training performance, ultimately, to find this out
    - But the other reason I want to avoid mode collapse is to get a representation that is comparable to the true low-dimensional state (comparable as in the same _sort_ of thing, not necessarily similar value). It would then also be interesting to manually vary the input to the decoder, or input actual low-dim states and compare the decoded frame to the actual frame
- Where to now
    - Keep increasing the latent dimension (up to say, 3 attempts) until mode collapse does not occur.
    - Modify the model to learn frame differences. I feel quite optimistic about this - there is so much redundant information in the frame, when only the movement of the object matters. Surely you can't get away with the equivalent solution to before - outputting zeros when loss pushes you to learn changes in form. If it does that then there is something else wrong with what I've built.

## 2019.12.24

Updating repo for gcloud port

- `conda env export --no-builds | grep -v "^prefix: " > environment.yml`
- Oh shoot, I forgot about MuJoCo. Can't use it on gcloud because I only have one license.
    - Oh wait. For now I just need MuJoCo images... I can transfer the dataset.
- Besides, the machine keeps shutting down without warning. I'll try it some other time.

Testing limit of mode collapse

- So currently we have filters [64, 64, 64, 1]
- Input shape is [128, 128, 3]
- So layer shapes go [128, 128, 3]->[64, 64, 64]->[32, 32, 64]->[16, 16, 64]->[8, 8, 1]
- Let's strip it right back: one layer
    - This gives [128, 128, 3]->[64, 64, 1]
    - So 4096 latent units. That's one twelfth of the original image.
- What if I'm wrong about the labels? What if the labels are the first image frame for some reason, and not tracking the input image? That would also explain why it learns something like the initial frame.
    - No, I can't see any evidence of that
- One layer, 5 epochs, 10k dataset
    - It tracks the state significantly! But why is it black and white?
- Two layer, 5 epochs, 10k dataset
    - Artefact-heavy
    - Generally does not track the state. Occasionally does. Of 15 test frams, I noticed one that tracked a significantly bent hopper.

## 2019.12.28

Generating `LunarLanderContinuous-v2` dataset

- Storing `.npz` of low-dim observations corresponding to frames
    - Modified dataset generation function significantly to do this. It seems to be OK.

## 2019.12.29

Inspecting `LunarLanderContinuous-v2` dataset

- Looks OK
- The visual rocket booster effects may confound things, but we'll see how it goes
- Compression artefacts are noticeable on the lander and vary somewhat randomly from frame to frame.

Update to plan

- I want to come back to the state autoencoder, but I'm going to take a slight tangent on something I think is probably easier: generating an image frame from the corresponding low-dim observation, or vice-versa. Respectively this is like training a CNN to be a rendering engine, or classify the underlying state based on a visual observation.
    - I anticipate a fair chance of blurring as a suboptimal solution to the former problem, which may warrant a GAN setup to improve. But GAN comes with its own challenges...

Setting up state->observation model training

- Input: "underlying" or "low-dimensional" observation - I will just refer to it here as "state"
- Output: rendered frame that correspond to state
- Loss: pixel-wise MSE
- With the current setup I don't expect many changes. It's nice that we have those TF data functions to easily switch out what the label is.
- What about the model? We should either write a new class or use the reference to the decoder portion of the autoencoder.
    - Let's subclass it
- Should we or shouldn't we have a Dense layer to begin? It may not be necessary. But it's the easiest way to make it compatible with the current interface.
- Testing training
    - Loss drops to about 5% in first epoch then kinda hits a wall. No further improvement in second epoch.

## 2019.12.30

Writing divorced Decoder class

- I think this will make things easier in the long run
- Ok, I've written a generic `SupervisedLearner` class, `ConvAutoencoder` class, and `ConvDecoder` class.
    - This raised some difficult design choices. For example, should the optimizer "belong" to the model or the learning algorithm? Put it that way and I'd think the learning algorithm, because it's the one performing optimization. On the other hand, in general there is an optimizer specific to each model, and a learning algorithm may have more than one model (e.g. GAN). I don't think there are gains in efficiency to be had here, it's just a matter of object-oriented principles and syntax. Which is better: `self.model.optimizer` or `self.model_optimizer`?
    - I'm straddling between the ideally abstract (which takes design effort that doesn't seem worth it for this project) and 100% specific (which repeats a lot of code).
    - The aim here is for `ConvDecoder` to only have to modify the model architecture and the source of data.

## 2020.01.01

Reviewing the OO interface

- It seems odd to set the random seed within the class. It seems more sensible to set it outside. But consider having a separate script that runs the class - maybe we wouldn't need to import numpy or tensorflow otherwise.

Testing decoder procedure

- I need to find the nearest square (even) power of 2 for resizing the input, not just the nearest square. Otherwise it won't scale up to 128x128.

## 2020.01.02

Adjusting dimensions of decoder

- On reflection I think it is a better idea to start with a Dense layer. The rationale of sparsity of convolution layers doesn't apply here because the input size is really small. For a length-8 observation vector and a size-64 Dense layer, we have 8x64=512 weights. By contrast, a size-64 convolution layer with kernel size 4 has 4x4x64=1024 weights. Granted, the Dense layer parameters have worse scale-up, but I don't expect it to be problematic.
- Should we normalise the observation data? I'm reluctant to use a sigmoidal activation in an initial Dense layer because of vanishing gradients.

Training StateObsLearner on LunarLander dataset

- 5 epochs
- Appear to flatten out at about 5.8% loss. This seems bad because I expect such loss can be achieved without accurately matching the position of the lunar lander.
- Next we need to write a test function, so we can visually inspect the outputs of the model.

## 2020.01.03

Writing test/visualisation method

Visualising predictions of lunar lander 5-epoch 10k sample decoder (seed: 20200101)

- It has essentially learned
    - An average of all backgrounds. There is a sort of central trapezium of solid white which is invariant in all the randomly generated terrains, with blurred grey blobs either side where the terrain varies.
    - To place the lander landing/crashing at a consistent location in some cases. This seems to correlate strongly with the lander actually hitting terrain at any location, and dimension 6 (from 0) in the low-dim observation being high value.
- I am not surprised by this result given the previous result for hopper.
- I guess it's time to mix in some adversarial loss. I expect this will give the decoder the guide it needs to find the more optimal solution, because the adversarial loss will both be less harsh on exact pixel-wise matching, and more harsh on this kind of "hedging" solution. On the other hand, adversarial loss tends to improve high-frequency detail, and may not be right to model the macro-details like position of objects.

## 2020.01.05

Testing 2 Dense layers

- Similar result
- Just realised that the low-dim state doesn't provide information about the terrain. Duh it can't model that!
- This strengthens the case for subtracting the background. Doing that now.
    - Bonus is smaller file size!

Testing background-subtracted dataset

- Subtracts the first frame of each episode
- Promising scenes: purple mist usually, roughly, in the right position. Sometimes a purple blob that is close to a blurred lander.
- Loss steadily improves
- Removing ReLU and making Dense activation 'tanh'
    - Same train loss, generally slightly worse eval loss
- 2 Dense (both ReLU)
    - About the same
    - Highly uncertain because one seed, 5 epochs, smaller dataset. Consider retrying on full scale setup.

Cloud machine time

- Transfer data with zip files - individual files take ages
- Ok gcloud rejects the ssh after a few minutes or less, with this error:
    ```
    ERROR: (gcloud.compute.ssh) [/usr/bin/ssh] exited with return code [255]
    ```
- SO 26193535 top answer is `sudo gcloud compute config-ssh` (without sudo for me)
    - It replies that no instances are running even when they are
- Was going to try another answer suggesting to create a new user with
    ```
    gcloud compute ssh newuser-name@instance-name
    ```
- But this said it couldn't find some property with mlpractical in the path
    - Which made me realise the 'project' property is still set to mlpractical project (use `gcloud config get-value project`)
    - Changed this to `instrumentality` with `gcloud config set project instrumentality`
- Experiment seems to run correctly, but seems to not use GPU
    - `tf.config.experimental.list_physical_devices('GPU')` prints nothing
    - `nvidia-smi` shows the P100 GPU is there
    - Maybe just need to `pip install tensorflow-gpu`
        - Success
- Problems with dataset ops on GPU
    - Interestingly, it works if I comment `@tf.function`. Not sure if this is supposed to be right.
- cuDNN version on machine and cuDNN compiled from TF source are different (7.4.1 vs. 7.6.0)
    - Solved this by installing `tensorflow-gpu-2.0.0b0` (I assume b1 was installed originally since this is the current default for plain `tensorflow`)
- 200 iterations per second ladies and gentlemen.

## 2020.01.06

Executing full-scale decoder experiment

- Still getting this error:
    ```
    ERROR: (gcloud.beta.compute.ssh) [/usr/bin/ssh] exited with return code [255].
    ```
- Quit

## 2020.01.07

Executing full-scale decoder experiment

- I stopped it at 80 epochs due to personal time constraints and because it was virtually converged.

## 2020.01.09

Analysing full-scale decoder experiment

- Firing up a notebook
- Why is `progress.txt` empty?
    - Could have been overwrited when I ran `model.test()` locally
    - I think we're safe because the original will still be on gcloud. But in general this is a bad state of affairs. We need to check whether the directory exists and confirm overwrite.
    - Changed Logger to confirm overwrite and only open output file when it is first needed.
- Plotting training data
    - Extremely smooth training curve
    - More rough validation curve
    - Validation loss indicates underfit for most of training, gradually moving to good fit by the end
- Plotting visual outputs
    - Learns position well
    - Lander is the right colour and roughly the right shape
    - Lander is too blurry to make out orientation

Visualising results in fresh environment simulation

- Model keeps putting the lander in the same position, about a quarter way below the top of frame
- Could it be the model is degenerate here?
    - I suspected I wasn't doing dataset shuffling right, so maybe it memorised the order of the data
    - But it doesn't change significantly from frame to frame either...
- Let's try plotting frames over larger spans of time to see if it sticks
    - It does kinda track the ground truth, but very poorly
- What if I wasn't matching observations to images properly?
    - But then unless I directly modified...oh! Normalisation? Yes...
    - `obs.mean()`: 0.06712762
    - `obs.std()`: 0.5819684
    - Phew. That's _much_ better. Tracks the ground truth very well.
- 
