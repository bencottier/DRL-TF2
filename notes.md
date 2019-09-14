# Notes for DPPG in Tensorflow 2.0

Author: Ben Cottier

## 2019.08.22

Working out best way to do target network ops

- Currently, the Model subclasses I have written are "lazy" in the sense that parameters are not initialised until the model is called.
- Current solution is to create dummy inputs, call the models, and then do the target ops. But I am not satisfied with the elegance of this solution.
- I have observed that parameters are also initalised if
    - I use the `Sequential` class to make the model
    - I use the functional API, e.g. `i = Input(...); x = Dense(...); model = Model(inputs=i, outputs=x)`
    - The above are elegant enough on their own, but don't fit with the way the class API is currently structured.
    - I think I am basically held back by calling `super()` in a `Model` subclass `__init__()` not-first. But this is fine. It's just that from experience, I usually try this and then switch back to calling `super()` first. I don't remember if there's a typical reason - maybe just that I forget about important stuff that is done in `super()` that needs to be done before other things I want to do.
        - Well, here's a reason from Keras itself against: 

            > RuntimeError: It looks like you are subclassing `Model` and you forgot to call `super(YourClass, self).__init__()`. Always start with this line.

    - Alternatively, I could test if just creating an `Input` object within the subclass does the job.
        - Nope!
    - We could _not_ subclass MLP on `Model`, and create the model within it using one of the methods listed above
        - Then we would lose the conveniences of subclassing, e.g. `call()`
            - Well, we could define it ourself using `__call__()`
    - How about using `tf.keras.Model.build`? Check the doc:

        > Builds the model based on input shapes received.
        > This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.
        > This method only exists for users who want to call model.build() in a standalone way (as a substitute for calling the model on real data to build it). It will never be called by the framework (and thus it will never throw unexpected errors in an unrelated workflow).

        - Yes, this is what we want. It won't save much code, but it is more elegant. I think we're done here.
- Target initialisation
    - I thought `tf.keras.models.clone_model` would serve, but this function copies the architecture, not the parameter values. So the initalisation is independent - not what we are after.
    - `model_target.set_weights(model.get_weights())` seems to do the trick.
    - Ok great, this is still working with the supposedly more elegant setup

## 2019.08.24

Writing target model assignment code

Writing `ReplayBuffer` class

- I was thinking of using a Python list, but given a 1M-sized buffer under cheetah would store 42M floats, I think numpy arrays are best. That means specifying each object that goes in the buffer and having a separate array for each object, so we lose generic-ness, but surely gain a lot in efficiency.

Writing main loop

- Copying in from SU DDPG code, and then converting the necessary bits
- The main part is the gradient updates

## 2019.08.26

Reducing EpochLogger to single-thread statistics

Debugging through implementation

- Fixed up use of `discount` as argument to `Critic` methods -- no longer an attribute, just passed to `backup` method
- Verified target update is correct for a single weight value for one layer.
    - TODO make sure that the order of weights in the list from `get_weights()` is invariant
- Fixed up to the point of running multiple epochs without error
- Logs are a go
- Checkpoints are a go
- Epochs are taking more than 3x longer than old implementation (over 5 minutes)...not good!
    - I want to blame TF2.0 but it's way too uncertain. Much could be problematic with my code.
    - Was the old implementation somehow using parallelism, and this is not?
    - This run even has 1 layers MLPs instead of 2, which would speed it up a little
    - Let's come at this fresh, and start with the obvious, simple ideas for what could be wrong.
    - What about moving the gradient tape calls out of the `for` loop, so they are only created once per epoch? No, I think you want to call it each time you compute a loss...
- On the bright side, after 10 epochs performance looks good.
- So there needs to be a proper investigation of speed. Once that's through, I would like to set the parameters identical SU exercise 2.2 and compare.

## 2019.08.27

Moving more actor-critic related code to class methods

Working out why this implementation is slower

- One clearly worthwhile thing to try is the `@tf.function` decorator above the heavy-duty functions, namely the training steps for actor and critic. This compiles the code into a graph and is designed to speed things up. But a 3x speedup? I am skeptical.
- Ok, I am killing the current cheetah run at 15 epochs (it was doing great in terms of performance - 3k average test return on epoch 15).
- Adding `tf.function` to `Actor.train_step` and `Critic.train_step`
    - ~200 seconds. A ~33% reduction, pretty good! But as expected, still nowhere near the 100 benchmark.
    - Oh, also, we aren't even controlling the parameters. Currently the neural nets are at least half the size!
- Controlling parameters to match TF1.0 benchmark
    - ~220 seconds.
- Hmm, given that it is warning about array ops and recommending `tf.where`, and I am just reading [here](https://www.tensorflow.org/beta/guide/autograph#batching) about using `tf.where` in place of list comprehension for-loops, I think `polyak_update` is likely another factor slowing it down. I predict this to be less significant than the training steps though...I'll stick my neck out and say a 10% reduction from the current 220 seconds, so about 200 seconds.
    - I wonder if `tf.where` is the best choice here though, because I don't have a condition on this for loop. To be fair, I'm not even certain that this is the function that causes the warning, but I guess we can find out quickly.
    - It still gives the warning if I remove code from `polyak_update()`
    - It prints in `ddpg.py` after `critic.train_step(...)`
    - This line: `gradients = tape.gradient(loss, self.trainable_variables)`
    - Ok, well I don't think we can do anything about it.
    - The warning is even printed out in an official TF2.0 example, so I feel ok
    - If I remove `polyak_update()` entirely, about 150 seconds. So updating the targets is a big chunk of the run time! But this doesn't counter my prediction yet - I need to use `tf.function`.
    - Ok, turns out that while `tf.assign` seems to be out of the 2.0 API, `Variable.assign` is alive and well. This seems to work well, and is perhaps even more elegant than the `[get,set]_weights()` method.
        - Throwing `tf.function` on this, and...165 seconds! That is a 20% error for my prediction. Impressive. Getting there, getting there...
- Put `tf.function` on `Actor.call()`: about 130 seconds. This is for the calls during experience collection - I think other calls are covered recursively by `train_step`. The surprises keep on coming!
- Now, it looks like I'm close to exhausting the easy speedups, but I want to check whether bundling the gradient tapes for actor and critic makes much difference. This will upset the class separation I just did today, but if the speedup is significant that's fine; there are good reasons to list the training process in the algorithm rather than the object (e.g. the training process is more a property of the algorithm that changes depending on DDPG, TD3, SAC etc., whereas actor-critic is a more consistent, basic idea).
    - Still about 130 seconds (to be exact, a 3 second speed up, but it's from a sample size of 1).
    - Well, good to know the object-oriented way is competetive here...but, I am actually favouring the in-algorithm option now, for the reason stated in the parent point. I'm anticipating changing the Actor-Critic classes anyway if I want them to be common to other algorithms (or perhaps it will be best to have DDPGCritic, TD3Critic and the like), but right now I _definitely_ anticipate the training steps to be tied to each algorithm.

Aside: the [Estimator](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/estimator/Estimator) API

- TF looks to be pushing this as a standard and core API for TF2.0
- It is an alternative to Keras
- I am willing to read up on this and move the implementation further in line with "the TF2.0 way", but it could take more time than I can afford in the coming period.

Profiling using `cProfile`

- Ok, first of all, I am running this in my native terminal and it's about 80 seconds...NOTE WELL, vscode debugger is for debugging, not runtime!
- The really baffling thing here is why `statistics_scalar` supposedly takes up about 30 seconds of the 80.
- Removing logger stats calls
- Yep, now it's 53 seconds. What is going on?
- Adding back `logger.store` calls, but not `log_tabular`
    - 55 seconds
- Ah, the mystery time is all spent converting Python lists to numpy arrays. This seems very solvable by modifying `EpochLogger` to use numpy arrays from the beginning. My guess is SpinningUp used lists because of compatibility with MPI, or just because the speedup from MPI is so great that it doesn't make much difference.

## 2019.08.27

Checking compatibility of logger using numpy arrays

- `statistics_scalar()` converts an object `x` to a numpy array
- `x` is `vals` in `log_tabular()`
- `vals` comes from

    ```python
    v = self.epoch_dict[key]
    vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
    ```

    - This means `vals` _would_ be a numpy array if `v` is a list of non-empty numpy arrays
- Ah, I see a reason why it is this way now. To use numpy arrays from the beginning, we would need to initialise it to some value, e.g. `zeros()`. But then to do the logs you need to know how many values to take statistics over.
    - This should be solved by a pointer attribute, incremented in `store()` and used to index and divide the statistics.
    - Actually, just a fixed `stat_size` would fit our purpose. It would be set to `steps_per_epoch`.
        - Wait no, different quantities have different frequencies of storage. Losses are stored every training step whereas return is every episode.
- So, it is more complicated than I first imagined - typical. But this array conversion takes up such a big fraction of runtime (35%) that I have a pretty high tolerance for added/changed code.
- Ok, I've done it, and it seems to work.
- 60 seconds folks. Amazing! And that's just on the debugger!
- Native terminal: 38.8 seconds. Astonishing. There could be some variation due to the state of the machine, but by direct comparison this is faster than when I commented out `log_tabular` (which gave 53 seconds). My immediate thought is that the `store` calls are also being significantly sped up now by assigning values in a pre-allocated numpy array instead of Python list appends.
- At any rate, this is superb speed up from when we started: about 5-fold. Using the native terminal I would expect a 3M-step cheetah run to take less than 7 hours.

Replicating run from SU exercise 2.2

- 1 hidden layer, 300 units
- 5000 steps per epoch, 5e4 steps -> 10 epochs
- 5000 start steps
- 150 ep len
- 64 batch size
- 0.95 polyak
- Seed 0, 10, 20
- Go!
- Ok, there is a bug in the new logging. Because train steps are performed all at once at the end of a trajectory, episode lengths that do not divide `steps_per_epoch` (e.g. 150 in 5000) cause a situation where some of the last trajectory in the previous epoch has not been trained on. That makes the array sizes for storing inconsistent. In this case, the pattern is 4950, 4950, 5100 recurring. How would I determine that pattern in advance?
    - 5000 / 150 = 33 r 50
    - 150 / 50 = 3 r 0
    - So the cycle is (33, 33, 33 + 1) x 150, or (5000-50, 5000-50, 5000+100)
    - Another example: ep_len = 350
        - 5000 / 350 = 14 r 100
        - 350 / 100 = 3 r 50
        - 350 / 50 = 7 r 0
        - The cycle is (14, 14, 14, 15, 14, 14, 15) x 350
        - 4900 = 5000 - (5000 % 350)
        - 5250 = 5000 + 350 - (5000 % 350)
        - Oh! Why don't we just set the size high enough so we don't have to compute this? It will waste a few zeros, but only a small bounded amount. It looks like the limit is `steps_per_epoch + max_ep_len - (steps_per_epoch % max_ep_len)`.
            - This seems to work, no significant slowdown
- Ok, running for real now: seeds 0, 10, 20
    - Very similar (as DRL run comparisons go)
    - Mine coasts around 200-300 in the last 3 epochs, SU grows steadily to 400-600 (this is consistent across seeds)
    - My Q loss is lower (seed avg. ~10 vs. ~20) and my pi loss is higher (seed avg. ~-200 vs. ~-400) in the final epoch
    - I am satisfied with the similarity, and I can't read too much into the trends given how little data there is.

Running DDPG HalfCheetah benchmark

- Parameters matched, good to go

## 2019.09.06

Plotting and reviewing benchmark results

- We have now run 10 seeds (0 to 90 in 10s)
- Plot command: `python -m spinup.run plot ./out/ddpg-benchmark-cheetah`
- Hmm, had trouble getting the plot to show something. Fixed by forcing `xaxis='Epoch'`, not sure why (normally it's `TotalEnvInteracts`). Anyway, the y axis is the same.
- Average performance is slightly higher than SU which is nice! SU stays below 6k (besides a few bumps near the end) while mine solidly clears it
- Variance is higher. Std. below is around 3300-3500 when stable, SU is 3700-3900. Std. above gets up to 9000-10000, SU 7000-8000.
- I should not read into this result as meaning my implementation is better, or different in any way inasmuch as it affects Return. The result is well explained by the usual variance between seeds for this configuration. I would say that 100 seeds for each version would settle whether there is any substantial difference.

## 2019.09.14

Working on a demo that is non-MuJoCo so it is free to run

- LunarLanderContinuous-v2 and MountainCarContinuous-v0 throw errors, not sure why
- Pendulum-v0
    - Bug
        - When act_dim is `1`, as it is here, `env.action_space.sample()` is `(1,)` while `get_action()` or the policy is `(1, 1)`.
        - As is, the switch to a `(1, 1)` shape after `start_steps` have elapsed causes `o2` to switch to `(obs_dim, 1)`, which does not fit the previously allocated buffer shape. Hence error.
        - If the `env.action_space.sample()` is reshaped to `(1, -1)` then the error occurs for `o2` straight up (before the switch) because the buffer has each element as 1D
        - If `get_action()` is squeezed on axis 0 then it is fixed but I would rather fix it on the `o2` end
        - Squeezing `o2` fixes it
        - Aside, it is odd that the error only happens for `act_dim == 1` because the switch from `(act_dim,)` to `(1, act_dim)` happens for any `act_dim`. Yet `o2` does not change shape after the switch for `act_dim > 1`.

Running Pendulum-v0 training

- `python main.py -s 0 --exp_name ddpg-demo-pendulum`

Preparing public repository
