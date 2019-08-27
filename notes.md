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

Aside: the [Estimator](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/estimator/Estimator) API

- TF looks to be pushing this as a standard and core API for TF2.0
- It is an alternative to Keras
- I am willing to read up on this and move the implementation further in line with "the TF2.0 way", but it could take more time than I can afford in the coming period.


