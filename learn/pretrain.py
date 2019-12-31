#!/usr/bin/env python
"""
pretrain.py

Model pretraining to enhance learning.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from algo.replay_buffer import ReplayBuffer
from model.autoencoder import ConvAutoencoder, ConvDecoder
from util.logger import EpochLogger
from util.utils import convert_json, scale_float, scale_uint8, setup_logger_kwargs
import tensorflow as tf
import numpy as np
import gym
import tqdm
import json
import PIL
import time
import pathlib
import os
import argparse


DATA_PATH = './data/state'
DATASET_SIZE = 10000  # 100000
AUTOTUNE = tf.data.experimental.AUTOTUNE


def generate_state_dataset(env_name, save_path, resume_from=0, 
    im_size=(128, 128), num_samples=int(1e5), max_ep_len=1000):

    output = json.dumps(convert_json(locals()), separators=(',',':\t'), indent=4, 
        sort_keys=True)
    save_path = os.path.join(save_path, env_name)
    obs_save_name = 'obs.npz'
    env = gym.make(env_name)

    if resume_from <= 0:
        obs_dim = env.observation_space.shape[0]
        obs = np.zeros(shape=(num_samples, obs_dim), dtype=np.float32)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), 'w') as out:
            out.write(output)
    else:
        print(f'Resuming from sample {resume_from}')
        obs = np.load(os.path.join(save_path, obs_save_name))['obs']

    print(f'State dataset of {num_samples:.0e} samples for {env_name} '
        f'saved to {save_path}')
    
    batch = -1
    sample = max(0, resume_from)
    save_path_base = save_path
    with tqdm.tqdm(total=num_samples, initial=sample) as pbar:  # create a progress bar
        while sample < num_samples:
            o, d, ep_len = env.reset(), False, 0
            while not (d or (ep_len == max_ep_len)):
                # Batch data so folders don't get too large
                this_batch = int(sample / 10000)
                if batch < this_batch:
                    batch = this_batch
                    save_path = os.path.join(save_path_base, f'data_batch_{batch}')
                    os.makedirs(save_path, exist_ok=True)

                # Store observation in array
                obs[sample] = o
                # Save observations
                np.savez_compressed(os.path.join(save_path_base, obs_save_name), obs=obs)
                # Save the frame as a downsampled RGB JPEG
                PIL.Image.fromarray(env.render(mode='rgb_array')).\
                    resize(im_size, resample=PIL.Image.BILINEAR).\
                    save(os.path.join(save_path, f'frame{sample}.jpg'), "JPEG")

                a = env.action_space.sample()
                o, _, d, _ = env.step(a)
                ep_len += 1
                sample += 1
                pbar.update(1)

                if sample >= num_samples:
                    break


def train_state_encoding(env_name, model_kwargs=dict(), seed=0, 
    epochs=100, lr=1e-3, batch_size=4, logger_kwargs=dict(), save_freq=1):
    """

    """
    # Set random seed for relevant modules
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Get observation dimensions
    with gym.make(env_name) as env:
        obs_dim = env.observation_space.shape[0]

    # Initialise dataset
    # tf.data helpers adapted from https://www.tensorflow.org/tutorials/load_data/images
    
    data_dir = os.path.join('./data/state', env_name)
    with open(os.path.join(data_dir, 'config.json')) as f:
        data_info = json.load(f)
    data_dir = pathlib.Path(data_dir)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

    @tf.function
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, data_info['im_size'])

    @tf.function
    def get_label(file_path):
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img

    @tf.function
    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    @tf.function
    def prepare_for_training(ds, cache=False, shuffle_buffer_size=1000):
        # Cache preprocessing work for dataset
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background 
        # while the model is training
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # Have multiple images loaded/processed in parallel
    ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # Split dataset into train and test
    train_size = int(0.9*DATASET_SIZE)
    test_size = DATASET_SIZE - train_size
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size).take(test_size)
    train_batches = int((train_size-1)/batch_size+1)
    test_batches = int((test_size-1)/batch_size+1)
    # Prepare datasets for iteration
    train_ds = prepare_for_training(train_ds, shuffle_buffer_size=train_size)
    test_ds = prepare_for_training(test_ds, shuffle_buffer_size=test_size)

    # Initialise model used for encoding
    input_batch, _ = next(iter(train_ds.take(1)))
    input_shape = input_batch.shape[1:]
    autoencoder = ConvAutoencoder(input_shape=input_shape,
        latent_dim=None, lr=lr, **model_kwargs)

    # Set up model checkpointing so we can resume training or test separately
    checkpoint_dir = os.path.join(logger_kwargs['output_dir'],
        'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    model_dict = {f'state_autoencoder': autoencoder}
    checkpoint = tf.train.Checkpoint(**model_dict)

    @tf.function
    def train_step(input_batch, label_batch):
        with tf.GradientTape(persistent=True) as tape:
            pred_batch = autoencoder(input_batch, training=True)
            loss = tf.keras.losses.mean_squared_error(label_batch, pred_batch)
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        autoencoder.optimizer.apply_gradients(
            zip(gradients, autoencoder.trainable_variables))
        return loss

    @tf.function
    def test_step(input_batch, label_batch):
        pred_batch = autoencoder(input_batch, training=True)
        loss = tf.keras.losses.mean_squared_error(label_batch, pred_batch)
        return loss

    def train(num_batch, ds, test=False):
        loss_name = 'train-loss' if not test else 'test-loss'
        step_fn = train_step if not test else test_step
        losses = np.zeros(num_batch, dtype=np.float32)
        with tqdm.tqdm(total=num_batch) as pbar:
            for i, (input_batch, label_batch) in enumerate(ds):
                loss = step_fn(input_batch, label_batch)
                losses[i] = np.sqrt(loss.numpy()).mean()
                pbar.update(1)
                pbar.set_description(
                    f'Epoch {epoch}: {loss_name}={losses[i]:.4f}')
            pbar.set_description(
                f'Epoch {epoch}: {loss_name}={losses.mean():.4f}')

    def test(num_batch, ds):
        train(num_batch, ds, test=True)

    for epoch in range(epochs):
        train(train_batches, train_ds)
        test(test_batches, test_ds)
        # Save the model
        if (epoch+1) % save_freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def test_pipeline():
    autoencoder = ConvAutoencoder([64, 64, 64, 1], 3)

    env_name = 'Hopper-v2'
    # env_name = 'Bowling-v0'
    env = gym.make(env_name)
    env.reset()
    im_frame = env.render(mode='rgb_array')

    print(im_frame.shape)

    import matplotlib.pyplot as plt

    im_pillow = PIL.Image.fromarray(im_frame)
    im_resize = im_pillow.resize((160, 160), resample=PIL.Image.BILINEAR)
    im = np.array(im_resize).astype(np.float32)
    im = scale_float(im)

    encoded_state = autoencoder(im[np.newaxis, ...], training=False)
    print(encoded_state)

    im_out = scale_uint8(autoencoder(im[np.newaxis, ...], training=True).numpy())

    plt.figure()
    plt.imshow(im_resize)
    plt.show()
    plt.figure()
    plt.imshow(im_out[0])
    plt.show()


def test_state_encoding(output_dir, env_name, checkpoint_number):
    # Get observation dimensions
    with gym.make(env_name) as env:
        obs_dim = env.observation_space.shape[0]
    
    autoencoder = ConvAutoencoder(input_shape=(128, 128, 3),
        latent_dim=None, hidden_sizes=(1,), kernel_size=4)

    checkpoint_dir = os.path.join(output_dir, 'training_checkpoints')
    checkpoint = tf.train.Checkpoint(state_autoencoder=autoencoder)
    if checkpoint_number is not None:
        checkpoint.restore(os.path.join(checkpoint_dir, f'ckpt-{checkpoint_number}')).expect_partial()
    else:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    import random
    import matplotlib.pyplot as plt

    for _ in range(5):
        rand_batch = random.choice(range(10))
        data_dir = os.path.join(DATA_PATH, env_name, f'data_batch_{rand_batch}')
        rand_img_name = random.choice(os.listdir(data_dir))
        rand_img = PIL.Image.open(os.path.join(data_dir, rand_img_name))

        x = scale_float(np.array(rand_img))
        decoded = autoencoder(x[np.newaxis, ...], training=True).numpy()

        plt.subplot(121)
        plt.imshow(rand_img)
        plt.subplot(122)
        plt.imshow(scale_uint8(decoded[0]))
        plt.show()


class SupervisedLearner(object):

    def __init__(self, epochs=100, batch_size=4, train_split=0.9, seed=0, 
        save_freq=1, logger_kwargs=dict(), data_kwargs=dict(), 
        model_kwargs=dict()):
        # Set random seed for relevant modules
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_split = train_split
        self.save_freq = save_freq
        self.input_shape = None
        self.logger_kwargs = logger_kwargs
        self.setup_dataset(**data_kwargs)
        self.setup_model(**model_kwargs)

    def setup_dataset_metadata(self):
        self.data_path = pathlib.Path('../data')
        self.data_info = dict()

    def setup_dataset(self, **kwargs):
        self.setup_dataset_metadata(**kwargs)
        list_ds = tf.data.Dataset.list_files(str(self.data_path/'*/*'))
        # Have multiple images loaded/processed in parallel
        ds = list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        # Split dataset into train and test
        train_size = int(self.train_split*DATASET_SIZE)
        test_size = DATASET_SIZE - train_size
        train_ds = ds.take(train_size)
        test_ds = ds.skip(train_size).take(test_size)
        self.train_batches = int((train_size - 1) / self.batch_size + 1)
        self.test_batches = int((test_size - 1) / self.batch_size + 1)
        # Prepare datasets for iteration
        self.train_ds = self.prepare_for_training(train_ds, 
            shuffle_buffer_size=train_size)
        self.test_ds = self.prepare_for_training(test_ds, 
            shuffle_buffer_size=test_size)
        # Determine input shape
        input_batch, _ = next(iter(self.train_ds.take(1)))
        self.input_shape = input_batch.shape[1:]

    def setup_model(self, **kwargs):
        self.model = tf.keras.Model()
        model_dict = {f'model': self.model}
        self.setup_model_checkpoint(model_dict)

    def setup_model_checkpoint(self, model_dict):
        # Set up model checkpointing so we can resume training or test separately
        checkpoint_dir = os.path.join(self.logger_kwargs['output_dir'],
            'training_checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(**model_dict)

    def get_input(self, file_path):
        raise NotImplementedError()

    def get_label(self, file_path):
        raise NotImplementedError()

    @tf.function
    def process_path(self, file_path):
        return self.get_input(file_path), self.get_label(file_path)

    @tf.function
    def prepare_for_training(self, ds, cache=False, shuffle_buffer_size=1000):
        # Cache preprocessing work for dataset
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(self.batch_size)
        # `prefetch` lets the dataset fetch batches in the background 
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def loss(self, label_batch, pred_batch):
        raise NotImplementedError()

    def postprocess_loss(self, loss):
        return loss

    @tf.function
    def train_step(self, input_batch, label_batch):
        with tf.GradientTape(persistent=True) as tape:
            pred_batch = self.model(input_batch, training=True)
            loss = self.loss(label_batch, pred_batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function
    def test_step(self, input_batch, label_batch):
        pred_batch = self.model(input_batch, training=True)
        loss = self.loss(label_batch, pred_batch)
        return loss

    def train_epoch(self, epoch, num_batch, ds, test=False):
        loss_name = 'train-loss' if not test else 'test-loss'
        step_fn = self.train_step if not test else self.test_step
        losses = np.zeros(num_batch, dtype=np.float32)
        with tqdm.tqdm(total=num_batch) as pbar:
            for i, (input_batch, label_batch) in enumerate(ds):
                loss = step_fn(input_batch, label_batch)
                losses[i] = self.postprocess_loss(loss)
                pbar.update(1)
                pbar.set_description(
                    f'Epoch {epoch}: {loss_name}={losses[i]:.4f}')
            pbar.set_description(
                f'Epoch {epoch}: {loss_name}={losses.mean():.4f}')

    def test_epoch(self, epoch, num_batch, ds):
        self.train_epoch(epoch, num_batch, ds, test=True)

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch, self.train_batches, self.train_ds)
            self.test_epoch(epoch, self.test_batches, self.test_ds)
            # Save the model
            if (epoch+1) % self.save_freq == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)


class StateAutoencoder(SupervisedLearner):

    def __init__(self, env_name, channels, model_kwargs=dict(), **kwargs):
        super(StateAutoencoder, self).__init__(**kwargs,
            data_kwargs=dict(env_name=env_name),
            model_kwargs=model_kwargs)
        self.channels = channels
        # Get observation dimensions
        with gym.make(env_name) as env:
            self.obs_dim = env.observation_space.shape[0]

    def setup_dataset_metadata(self, env_name):
        self.data_path = pathlib.Path('../data/state')
        self.data_path /= env_name
        with open(str(self.data_path/'config.json')) as f:
            self.data_info = json.load(f)

    def setup_model(self, **kwargs):
        self.model = ConvAutoencoder(input_shape=self.input_shape, 
            latent_dim=self.obs_dim, **kwargs)
        model_dict = {f'state_autoencoder': self.model}
        self.setup_model_checkpoint(model_dict)

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=self.channels)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, self.data_info['im_size'])

    def get_input(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img

    def loss(self, label_batch, pred_batch):
        return tf.keras.losses.mean_squared_error(label_batch, pred_batch)

    def postprocess_loss(self, loss):
        return np.sqrt(loss.numpy()).mean()


class StateDecoder(SupervisedLearner):

    def __init__(self, env_name, channels, model_kwargs=dict(), **kwargs):
        super(StateAutoencoder, self).__init__(**kwargs,
            data_kwargs=dict(env_name=env_name),
            model_kwargs=model_kwargs)
        self.channels = channels
        # Get observation dimensions
        with gym.make(env_name) as env:
            self.obs_dim = env.observation_space.shape[0]

    def setup_dataset_metadata(self, env_name):
        self.data_path = pathlib.Path('../data/state')
        self.data_path /= env_name
        with open(str(self.data_path/'config.json')) as f:
            self.data_info = json.load(f)
        self.obs = np.load(str(self.data_path/'obs.npz'))['obs']

    def setup_model(self, **kwargs):
        self.model = ConvDecoder(input_shape=self.input_shape, **kwargs)
        model_dict = {f'state_decoder': self.model}
        self.setup_model_checkpoint(model_dict)

    def get_input(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        name = tf.strings.split(parts, '.')[0]
        idx = tf.strings.substr(name, 5, 10)
        idx = tf.strings.to_number(idx, tf.int32)
        idx = int(idx.numpy())
        o = self.obs[idx]
        return o


if __name__ == '__main__':
    # test_pipeline()

    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
            help='environment name (from OpenAI Gym)')

    parser.add_argument('--dataset', action="store_true",
            help='generate a dataset of states for the environment'
                'instead of training')
    parser.add_argument('--resume', type=int, default=0,
            help='data sample index to resume generation from')

    parser.add_argument('--test_dir', type=str, default=None,
            help='directory containing training_checkpoints folder '
                '(triggers test mode)')
    parser.add_argument('--checkpoint', type=int, default=None,
            help='checkpoint to load models from (default latest)')

    parser.add_argument('--hid', type=int, default=64,
            help='number of hidden units per hidden layer')
    parser.add_argument('--l', type=int, default=3,
            help='number of hidden layers')
    parser.add_argument('--seed', '-s', type=int, default=0,
            help='random seed')
    parser.add_argument('--epochs', type=int, default=50,
            help='number of epochs to train')
    parser.add_argument('--exp_name', type=str, default='unnamed',
            help='name for this experiment, used for the logging folder')
    args = parser.parse_args()

    if args.dataset:
        generate_state_dataset(args.env, DATA_PATH, 
            resume_from=args.resume)
    elif args.test_dir is not None:
        test_state_encoding(args.test_dir, args.env, args.checkpoint)
    else:
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
        hidden_sizes = args.l*[args.hid]+[1]
        train_state_encoding(args.env, seed=args.seed, epochs=args.epochs,
            model_kwargs=dict(hidden_sizes=hidden_sizes, kernel_size=4),
            logger_kwargs=logger_kwargs)
