"""
example.py

Playground to test TensorFlow 2.0 example code.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from mlp import MLP
import tensorflow as tf


def minimal_train():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test)


def model_weights():
    model1 = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    model2 = MLP((400, 300), 'relu')
    i = tf.keras.Input(2,)
    x = tf.keras.layers.Dense(5)(i)
    model3 = tf.keras.Model()#inputs=i)#, outputs=x)

    print(model1.trainable_weights)
    print(model2.trainable_weights)
    print(model3.trainable_weights)


def model_copying():
    # model1 = tf.keras.Model()
    i = tf.keras.Input((2, 3))
    x = tf.keras.layers.Dense(2)(i)
    model1 = tf.keras.Model(inputs=i, outputs=x)
    # model1 = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
    #         tf.keras.layers.Dense(128, activation='relu'),
    #         tf.keras.layers.Dropout(0.2),
    #         tf.keras.layers.Dense(10, activation='softmax')
    #     ])
    model2 = tf.keras.models.clone_model(model1)
    
    print(model1.trainable_weights)
    print(model2.trainable_weights)


def model_implicit_layers():
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            # self.input1 = tf.keras.layers.Input(shape=(2, 3))
            self.input1 = tf.keras.Input(shape=(2, 3))
            self.dense1 = tf.keras.layers.Dense(2)

    model = MyModel()
    print(model.layers)
    print(model.trainable_weights)


if __name__ == '__main__':
    # minimal_train()
    # model_weights()
    # model_copying()
    model_implicit_layers()
