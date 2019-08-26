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

- 
