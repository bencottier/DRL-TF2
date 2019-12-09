#!/usr/bin/env python
"""
pretrain.py

Model pretraining to enhance learning.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from replay_buffer import ReplayBuffer
from logger import EpochLogger
from autoencoder import ConvolutionalAutoencoder
from utils import convert_json, scale_float, scale_uint8
import tensorflow as tf
import numpy as np
import gym
import tqdm
import json
import PIL
import time
import os


def generate_state_dataset(env_name, save_path, resume_from=0, 
    im_size=(128, 128), num_samples=int(1e5), max_ep_len=1000):

    if resume_from <= 0:
        output = json.dumps(convert_json(locals()), separators=(',',':\t'), indent=4, 
            sort_keys=True)
        save_path = os.path.join(save_path, env_name)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.json"), 'w') as out:
            out.write(output)
    else:
        save_path = os.path.join(save_path, env_name)
        print(f'Resuming from sample {resume_from}')

    print(f'State dataset of {num_samples:.0e} samples for {env_name} '
        f'saved to {save_path}')

    batch = 0
    env = gym.make(env_name)
    sample = max(0, resume_from)
    save_path_base = save_path
    with tqdm.tqdm(total=num_samples, initial=sample) as pbar:  # create a progress bar
        while sample < num_samples:
            _, d, ep_len = env.reset(), False, 0
            while not (d or (ep_len == max_ep_len)):
                # Batch data so folders don't get too large
                this_batch = int(sample / 10000)
                if batch < this_batch:
                    batch = this_batch
                    save_path = os.path.join(save_path_base, f'data_batch_{batch}')
                    os.makedirs(save_path, exist_ok=True)

                # Save the frame as a downsampled RGB JPEG
                PIL.Image.fromarray(env.render(mode='rgb_array')).\
                    resize(im_size, resample=PIL.Image.BILINEAR).\
                    save(os.path.join(save_path, f'frame{sample}.jpg'), "JPEG")

                a = env.action_space.sample()
                _, _, d, _ = env.step(a)
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

    # Initialise model used for encoding
    autoencoder = ConvolutionalAutoencoder(lr=lr, **model_kwargs)

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
    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # Cache preprocessing work for dataset
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background 
        # while the model is training
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # Have multiple images loaded/processed in parallel
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = labeled_ds.shuffle(buffer_size=1000)
    # Split dataset into train and test
    ds_size = 50000 # TODO
    train_size = int(0.8*ds_size)
    test_size = ds_size - train_size
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size).take(test_size)
    train_batches = int((train_size-1)/batch_size+1)
    test_batches = int((test_size-1)/batch_size+1)
    # Prepare datasets for iteration
    train_ds = prepare_for_training(train_ds, cache='./states_train.tfcache')
    test_ds = prepare_for_training(test_ds, cache='./states_test.tfcache')

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
        pred_batch = autoencoder(input_batch, training=False)
        loss = tf.keras.losses.mean_squared_error(label_batch, pred_batch)
        return loss

    for epoch in range(epochs):
        with tqdm.tqdm(total=train_batches) as pbar_train:
            for input_batch, label_batch in train_ds:
                loss = train_step(input_batch, label_batch)
                loss = loss.numpy().mean()
                pbar_train.update(1)
                pbar_train.set_description(f'Epoch {epoch}: train-loss={loss:.4f}')
        with tqdm.tqdm(total=test_batches) as pbar_test:
            for input_batch, label_batch in test_ds:
                loss = test_step(input_batch, label_batch)
                loss = loss.numpy().mean()
                pbar_test.update(1)
                pbar_test.set_description(f'Epoch {epoch}: test-loss={loss:.4f}')
        # Save the model
        if (epoch+1) % save_freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def test_pipeline():
    autoencoder = ConvolutionalAutoencoder([64, 64, 64, 1], 3)

    env_name = 'Hopper-v2'
    # env_name = 'Bowling-v0'
    env = gym.make(env_name)
    o = env.reset()
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


if __name__ == '__main__':
    # test_pipeline()

    generate_state_dataset('Hopper-v2', './data/state')
