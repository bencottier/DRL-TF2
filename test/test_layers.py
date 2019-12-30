import tensorflow as tf

latent_dim = 11

H = 128
W = 128
C = 3
hidden_sizes = (64, 64, 64, 1)
restore_shape = (int(H/(2**len(hidden_sizes))), int(W/(2**len(hidden_sizes))), 1)
restore_units = restore_shape[0] * restore_shape[1]

inputs = tf.keras.Input(shape=(H, W, C))
x = inputs

for h in hidden_sizes:
    x = tf.keras.layers.Conv2D(h, 3, strides=2, padding='same')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
encoder_outputs = x
x = tf.keras.layers.Dense(restore_units, activation='relu')(x)
x = tf.keras.layers.Reshape(restore_shape)(x)
x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)

outputs = x
model = tf.keras.Model(inputs=inputs, outputs=outputs)
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
