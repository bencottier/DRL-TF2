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

## 2019.11.01

Warning

- Recently I tried running some environment and got an error with logging. I think it was a bug that I thought I had fixed, but apparently not. Unfortunately I don't remember which environment I was trying to run on, but trying a few will probably uncover it.

Reviewing TD3 algorithm

- I will write in terms of differences to DDPG
- Target policy smoothing
    - Add clipped (Gaussian) noise to target policy action, and clip the result to be within action limits.
    - Noise clipping limits and noise std are hyperparameters
    - What this does: a form of regularisation. Suppose the Q-function approximator happens to have a sharp park (i.e. a peak over a small subset of the action space). Adding noise to the action means that similar actions (small Euclidean distance in action space) are less distinguished, and thus the peak is less likely to be exploited.
    - Relating this to our DDPG implementation, we would add noise to `actor_target` before it gets passed to `critic_target` for `q_pi_targ`. Not to be confused with the `act_noise` added to the current (non-target) policy.
- Clipped double-Q learning ("Twin")
    - Have two Q-function estimators (both regular and target)
    - When computing Bellman backup, use the minimum of the two Q targets
    - Update both regular Q estimators using this clipped backup
    - Research question: generalising from 2 to N, how does the benefit vary?
        - Would check TD3 paper to see if they investigated this at all
- Less frequent policy updates ("Delayed")
    - Already implemented in our DDPG (1 policy update per trajectory)
    - Can further reduce policy update frequency by multiple trajectories (SpinningUp default is 2)
    - Note Q update frequency is kept to 1

## 2019.11.09

- Which critic is used to update the policy? Presumably the minimum one
    - Oh, apparently you just use Q1 by convention. Which indicates that it doesn't really matter for the policy's learning.
- Aside: I feel like there is some natural extension to the idea of TD3. Not simply increasing the number of Q estimators, but the fact that the two Q estimators ultimately differ in terms of their random initialisation (because they get the same weight updates thereafter). What could we do with the initialisation that might amplify the benefit of this method?
- I'm thinking there may be a way to stack the q values and do some of the operations without a python for loop, but given Q is conventionally 2 and I'm not going to test exceedingly high numbers of Q networks, I think the efficiency gains are not worth the overhead at this point
- What to log?
    - Settling on mean and std of the Q estimator values and losses

## 2019.11.17

Where we are at

- So, I wasn't clear that last time I was extending TD3 to take the number of Q estimators as an argument, in case I want to experiment with that.
    - Not sure if I tested this update - IIRC did a quick test for two Q networks (i.e. standard TD3, but as one case of the n-critic implementation)
    - 3 critics seems OK (10 epochs on `Pendulum-v0`)
    - 1 critic ditto

Investigating logging bug

- We discovered this some time ago but neglected it because it didn't affect `Pendulum-v0` or `HalfCheetah-v2`. I think it may have affected `Swimmer` or `LunarLander-Continuous`, but I'll have to check.
- The bug may not appear until the first, second, even the third epoch, so need patience.
- `Swimmer-v2`, DDPG
    - 5 epochs OK
- `LunarLanderContinuous-v2`, DDPG
    - `AttributeError: module 'gym.envs.box2d' has no attribute 'LunarLanderContinuous'`
    - But when `LunarLanderContinuous-v1`: `gym.error.DeprecatedEnv: Env LunarLanderContinuous-v1 not found (valid versions include ['LunarLanderContinuous-v2'])`
    - Just to be clear, this isn't the bug I'm looking for
    - https://github.com/openai/gym/issues/1603 - "Seems to be with the prelim version, installing full version with pip install 'gym[all]' solved it."
    - Bingo: occurs before first epoch end

        ```
        Traceback (most recent call last):
        File "train.py", line 39, in <module>
            algos[args.algo.lower()](lambda : gym.make(args.env), logger_kwargs, args)
        File "/home/ben/projects/drl-tf2/ddpg.py", line 254, in run
            steps_per_epoch=10000, logger_kwargs=logger_kwargs)
        File "/home/ben/projects/drl-tf2/ddpg.py", line 222, in ddpg
            logger.store(max_logger_steps // max_ep_len, EpRet=ep_ret, EpLen=ep_len)
        File "/home/ben/projects/drl-tf2/logger.py", line 219, in store
            self.epoch_dict[k][0][self.epoch_dict[k][1]] = v
        IndexError: index 10 is out of bounds for axis 0 with size 10
        ```

- `BipedalWalker-v2`, DDPG
    - Identical error to `LunarLanderContinuous-v2` (including index 10)
- `MountainCarContinuous-v0` OK 
- `Hopper-v2`
    - Identical error to `LunarLanderContinuous-v2` (including index 10)
- This is what I said about the bug which I don't think I fully solved:

    > Ok, there is a bug in the new logging. Because train steps are performed all at once at the end of a trajectory, episode lengths that do not divide `steps_per_epoch` (e.g. 150 in 5000) cause a situation where some of the last trajectory in the previous epoch has not been trained on. That makes the array sizes for storing inconsistent. In this case, the pattern is 4950, 4950, 5100 recurring. How would I determine that pattern in advance?

- Information
    - The number 10 comes from argument `shape` to `EpochLogger.store()`
        - `shape` is passed as `max_logger_steps // max_ep_len`
        - `max_logger_steps` is `steps_per_epoch`, 10000
        - `max_ep_len` is 1000
    - `self.epoch_dict[k][0]` has shape `(10,)`
    - `self.epoch_dict[k][1]` is 10
    - We are trying to index the former with the latter
    - This entails that `store` is being called more times than the expected shape of 10
    - Let's see how high the index for `EpRet` goes if we comment this out
        - 0 to 433 inclusive
        - So it calls 434 times
        - This seems a bizarre number
        - Epoch 2: 232 times
        - Epoch 3: 101 times
        - Epoch 4: 42 times
        - Epoch 5: 9 times
        - Ok, see, this method gets called for `EpRet` if `d or (ep_len == max_ep_len)`. The `d` is key: if it reaches a terminal state. This depends on the agent's behaviour.
            - Note that `LunarLander` varies in the timing of termination.
            - I predict Hopper is too...yes: `done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))`.
            - Whereas `HalfCheetah-v2` has `done=False`.
            - This is also consistent with the number of calls decreasing each epoch: the agent is learning to avoid terminating early.
    
Solution

- This confounds our fixed-sized array logging. Now I see why SpinningUp was using lists. But I think the time gains are really valuable. What to do?
- With early termination as a possibility, these episode-wise logs could be anywhere from `max_logger_steps // max_ep_len` to `max_logger_steps` in length.
- So space efficiency will take a hit, but not much in absolute terms, and I'm much more concerned with time efficiency here. I think it's acceptable to give the arrays maximum size.
- We can handle non-full arrays using the index `self.epoch_dict[k][1]`
    - Actually, I think I already do this
- Note that, unless episode length is exactly in line with max episode length, the experience of the final episode of an epoch is only learned at the start of the next epoch. This is only consequential in the final epoch, where some experience will be wasted. However it's not very consequential anyway; the experience is very likely negligible amidst a long training period.
    - Nonetheless I think this is addressed by adding `or (t == total_steps-1)` to the learning condition

More problems

- Ok...have a problem now with 'QMeans' (may not be the only one) hitting index 10000. This is the number of steps per epoch, so currently I can't conceive how this is possible even with variable termination. 
    - Note this is at the 3rd epoch, so may depend on randomness
    - It's not the new final-epoch condition I added, because `t` is 29609 and `total_steps` is 30000
    - Sum of episode lengths is 9994, and this is yet to be updated in this iteration. The current episode length is 172, which would exceed 10000.
        - Ok, I see the problem. But what exactly are the conditions for this case, i.e. why does it not always occur?
        - Ah, remember we talked about the final episode from the previous epoch possibly carrying over to the next. So what's happened is (previous_epoch_ep_len + sum(current_epoch_ep_lens) >= steps_per_epoch - ep_len).
        - So we could change this new, third condition to AND, with the opposite of the above - should do internally to `EpochLogger`
- Ok, successful 3-epoch run, but longer runs and varied `max_ep_len` needed to properly validate

## 2019.11.18

Testing logger fix further

- 50-epoch run OK
- 50-epoch run with episode length 157 (just to be unusual)
    - Training episode length tends to be longer - first guess is this is caused by the new condition on the gradient update/logging block.
    - Don't have time to confirm now but, probably better to remove that condition and instead make the size of the buffers large enough to handle the case where there would otherwise be not enough room

## 2019.11.27

- So, we had a problem where `(previous_epoch_ep_len + sum(current_epoch_ep_lens) >= steps_per_epoch - ep_len)`
- I attempted to avoid this with the condition `self.epoch_dict[key][0].sum() + length < max_size`
    - `self.epoch_dict[key][0].sum()` is `(previous_epoch_ep_len + sum(current_epoch_ep_lens)`
    - `length` is `ep_len`
    - `max_size` is `max_logger_steps` which is equal to `steps_per_epoch` or `steps_per_epoch + max_ep_len - (steps_per_epoch % max_ep_len)`
- We then had an issue with variable episode length _above_ the nominal maximum episode length
- Propose to remove the new condition and instead set the maximum size of the logger array such that the condition never applies
- Given we didn't accoun for terminal states, I think a solution is to set `max_logger_steps = steps_per_epoch + max_ep_len` regardless of whether `max_ep_len` divides `steps_per_epoch`. 
- Note MuJoCo license appears to have just expired.
- Testing `LunarLanderContinuous-v2` with episode length 157
    - Hmm I'm getting apparently unrelated problems now...
  
## 2019.11.28

- We made a tentative fix to the logging length issue, but then encountered an apparently unrelated error when running `LunarLanderContinuous-v2` with episode length 157
- Let's replicate and copy the error message

    ```
    Traceback (most recent call last):
      File "train.py", line 39, in <module>
        algos[args.algo.lower()](lambda : gym.make(args.env), logger_kwargs, args)
      File "/home/ben/projects/drl-tf2/td3.py", line 275, in run
        steps_per_epoch=10000, logger_kwargs=logger_kwargs)
      File "/home/ben/projects/drl-tf2/td3.py", line 253, in td3
        test_agent()
      File "/home/ben/projects/drl-tf2/td3.py", line 182, in test_agent
        o, r, d, _ = test_env.step(get_action(o, 0))
      File "/home/ben/miniconda3/envs/drl-tf2/lib/python3.7/site-packages/gym/wrappers/time_limit.py", line 15,         in step
        observation, reward, done, info = self.env.step(action)
      File "/home/ben/miniconda3/envs/drl-tf2/lib/python3.7/site-packages/gym/envs/box2d/lunar_lander.py", line         250, in step
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    ```

- So this is a problem with action not being zero-dimensional
- So `get_action` should return a zero-dimensional array IF there is only one element? I fear this will upset other interfaces, but let's look into it.
    - When the error occurs, `t` is 9999 and `get_action(o, 0)` has shape (2,) (consistent in three runs)
    - Values in three runs: [-1., -1.], [-1., -1.], [-0.9999998, 0.79131085]
    - `act_dim` is 2 and `obs_dim` is 8
    - Oh, this is occuring in `test_agent`. So `t` is not so relevant.
    - Odd, I said action shape came out as (2,) the first time, but since have found (1, 2)
- Actually, the problem may be with the continuous flag. Notice the code that raises. If it's `self.continuous` (which it should be), it assumes the action has more than one element. 
    - Let's check the `test_env.env.continuous` flag...it's `True`
        - But, action[0] would evaluate to False - so it would move on to the next condition
        - But then given self.continuous is True, shouldn't Python evalute the condition before reaching `action==2`?
        - Let's step into the code
        - Ok step into doesn't work
- Aha: because it's shape `(1, 2)`, `action[0]` gives an array of shape (2,). So it is failing on the first part of the condition after all. 
    - This relates back to the actor network outputting shape `(1, act_dim)` while `env.action_space.sample()` outputs `(act_dim,)` - note that at the time of the error, we are still in the latter regime for training
    - But I recall having problems with reshaping the actor output to `(act_dim,)` - would need to check multiple environments work with this, if any
        - See 2019.09.14
        - "If `get_action()` is squeezed on axis 0 then it is fixed but I would rather fix it on the `o2` end"
        - Ok, not sure why I rathered that
    - Is the issue caused related to having an effective batch size of 1 for testing?
    - Squeezing `get_action()` gets past the error now
    - `LunarLanderContinuous-v2` 5 epochs OK
    - Let's try on other envs
        - `Pendulum-v0` OK
        - `BipedalWalker-v2` OK
        - `MountainCarContinuous-v0` OK


