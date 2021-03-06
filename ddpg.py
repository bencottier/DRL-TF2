#!/usr/bin/env python
"""
ddpg.py

Main script for the deep deterministic policy gradient (DDPG) algorithm.

DDPG was first proposed by [Lillicrap et al.](https://arxiv.org/abs/1509.02971) 

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from mlp import MLP
from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer
from logger import EpochLogger
import tensorflow as tf
import numpy as np
import time
import os


def ddpg(env_fn, ac_kwargs=dict(), seed=0, steps_per_epoch=5000, epochs=100, 
         replay_size=int(1e6), discount=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, 
         batch_size=100, start_steps=10000, act_noise=0.1, max_ep_len=1000, 
         logger_kwargs=dict(), save_freq=1):
    """
    Implements the deep deterministic policy gradient algorithm.

    Performance statistics are logged to stdout and to file in 
    CSV format, and models are saved regularly during training.

    Args:
        env_fn: callable. Must load an instance of an environment 
            that implements the OpenAI Gym API.
        ac_kwargs: dict. Additional keyword arguments to be passed 
            to the Actor and Critic constructors.
        seed: int. Random seed.
        steps_per_epoch: int. Number of training steps or 
            environment interactions that make up one epoch.
        epochs: int. Number of epochs for training.
        replay_size: int. Maximum number of transitions that 
            can be stored in the replay buffer.
        discount: float. Rate of discounting on future reward, 
            usually denoted with the Greek letter gamma. Normally 
            between 0 and 1.
        polyak: float. Weighting of target estimator parameters
            in the target update (which is a "polayk" average).
        pi_lr: float. Learning rate for the policy or actor estimator.
        q_lr: float. Learning rate for the Q or critic estimator.
        batch_size: int. Number of transitions to sample from the 
            replay buffer per gradient update of the estimators.
        start_steps: int. Number of initial training steps where 
            actions are chosen at random instead of the policy, 
            as a means of increasing exploration.
        act_noise: float. Scale (standard deviation) of the Gaussian 
            noise added to the policy for exploration during training.
        max_ep_len: int. Maximum number of steps for one episode in 
            the environment. Episode length may be shorter if there
            are terminal states.
        logger_kwargs: dict. Keyword arguments to be passed to the 
            logger. Can be set up using utils.setup_logger_kwargs().
        save_freq: int. Models are saved per this number of epochs.
        
    """
    # Set up logging
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set random seed for relevant modules
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Create environment
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if env._max_episode_steps < max_ep_len:
        max_ep_len = env._max_episode_steps
    if steps_per_epoch % max_ep_len != 0:
        """
        Training steps are batched at the end of a trajectory, so if 
        episode length does not divide steps per epoch, the size of 
        training step log arrays can be inconsistent. This takes the 
        upper bound on size, which wastes some memory but is easy.
        """
        max_logger_steps = steps_per_epoch + max_ep_len - (steps_per_epoch % max_ep_len)
    else:
        max_logger_steps = steps_per_epoch

    # Action limit for clipping
    # Assumes all dimensions have the same limit
    act_limit = env.action_space.high[0]

    # Give actor-critic model access to action space
    ac_kwargs['action_space'] = env.action_space

    # Randomly initialise critic and actor networks
    critic = Critic(input_shape=(batch_size, obs_dim + act_dim), lr=q_lr, **ac_kwargs)
    actor = Actor(input_shape=(batch_size, obs_dim), lr=pi_lr, **ac_kwargs)

    # Initialise target networks with the same weights as main networks
    critic_target = Critic(input_shape=(batch_size, obs_dim + act_dim), **ac_kwargs)
    actor_target = Actor(input_shape=(batch_size, obs_dim), **ac_kwargs)
    critic_target.set_weights(critic.get_weights())
    actor_target.set_weights(actor.get_weights())

    # Initialise replay buffer for storing and getting batches of transitions
    replay_buffer = ReplayBuffer(obs_dim, act_dim, size=replay_size)

    # Set up model checkpointing so we can resume training or test separately
    checkpoint_dir = os.path.join(logger.output_dir, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(critic=critic, actor=actor)

    def get_action(o, noise_scale):
        """
        Computes an action from the policy (as a function of the 
        observation `o`) with added noise (scaled by `noise_scale`),
        clipped within the bounds of the action space.
        """
        a = actor(o.reshape(1, -1))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    @tf.function
    def train_step(batch):
        """
        Performs a gradient update on the actor and critic estimators
        from the given batch of transitions.

        Args:
            batch: dict. A batch of transitions. Must store valid 
                values for 'obs1', 'acts', 'obs2', 'rwds', and 'done'. 
                Obtained from ReplayBuffer.sample_batch().
        Returns:
            A tuple of the Q values, critic loss, and actor loss.
        """
        with tf.GradientTape(persistent=True) as tape:
            # Critic loss
            q = critic(batch['obs1'], batch['acts'])
            q_pi_targ = critic_target(batch['obs2'], actor_target(batch['obs2']))
            backup = tf.stop_gradient(batch['rwds'] + discount * (1 - batch['done']) * q_pi_targ)
            q_loss = tf.reduce_mean((q - backup)**2)
            # Actor loss
            pi = actor(batch['obs1'])
            q_pi = critic(batch['obs1'], pi)
            pi_loss = -tf.reduce_mean(q_pi)
        # Q learning update
        critic_gradients = tape.gradient(q_loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))
        # Policy update
        actor_gradients = tape.gradient(pi_loss, actor.trainable_variables)
        actor.optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
        return q, q_loss, pi_loss

    def test_agent(n=10):
        """
        Evaluates the deterministic (noise-free) policy with a sample 
        of `n` trajectories.
        """
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(n, TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    for t in range(total_steps):
        """
        Start with `start_steps` number of steps with random actions,
        to improve exploration. Then use the learned policy with some 
        noise added to keep up exploration (but less so).
        """
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Execute a step in the environment
        o2, r, d, _ = env.step(a)
        o2 = np.squeeze(o2)  # bug fix for Pendulum-v0 environment, where act_dim == 1
        ep_ret += r
        ep_len += 1
        
        """
        Ignore the "done" signal if it comes from hitting the time
        horizon (that is, when it's an artificial terminal signal
        that isn't based on the agent's state)
        """
        d = False if ep_len==max_ep_len else d

        # Store transition in replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Advance the stored state
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)

                # Actor-critic update
                q, q_loss, pi_loss = train_step(batch)
                logger.store((max_logger_steps, batch_size), QVals=q.numpy())
                logger.store(max_logger_steps, LossQ=q_loss.numpy(), LossPi=pi_loss.numpy())

                # Target update
                critic_target.polyak_update(critic, polyak)
                actor_target.polyak_update(actor, polyak)

            logger.store(max_logger_steps // max_ep_len, EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Post-training for this epoch: save, test and write logs
        if t > 0 and (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save the model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                checkpoint.save(file_prefix=checkpoint_prefix)

            # Test the performance of the deterministic policy
            test_agent()

            # Log info about the epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t+1)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


def run(env_fn, logger_kwargs, args):
    ddpg(env_fn, seed=args.seed, discount=args.discount,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
            epochs=args.epochs, batch_size=100,
            steps_per_epoch=10000, logger_kwargs=logger_kwargs)
