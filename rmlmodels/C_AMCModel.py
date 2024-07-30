
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,Dense,Conv1D,GlobalAveragePooling1D,ReLU,BatchNormalization,Dropout,Softmax
from keras.layers import LSTM,MultiHeadAttention,Flatten,LSTM,Reshape,Bidirectional,Layer
from prunable_layers import PruneBidirectional
from tensorflow.keras.optimizers import Adam
from keras import backend as K


class SumLayer(Layer):
    def __init__(self, **kwargs):
        super(SumLayer, self).__init__(**kwargs)

    def call(self, inputs):
        summed_output = tf.reduce_sum(inputs, axis=1)
        return summed_output

class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, snr, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.snr = snr

    def call(self, x, training=None):
        noise_shape = tf.shape(x)
        mean = 0.0
        std_dev = 1.0
        random_noise = tf.random.normal(noise_shape, mean=mean, stddev=std_dev)
        random_noise_energy = tf.reduce_sum(tf.abs(random_noise) ** 2)
        signal_energy = tf.reduce_sum(tf.square(x))
        signal_energy_expected = random_noise_energy * (10 ** (self.snr / 10.0))
        normalized_x = x * tf.sqrt(signal_energy_expected) / tf.sqrt(signal_energy)
        total_vector = normalized_x + random_noise
        return total_vector
    def get_config(self):
        config = super(GaussianNoiseLayer, self).get_config()
        config.update({"snr": self.snr})
        return config

def Model(weights = None,
              input_shape=[512, 2],
              classes = 11,
              **kwargs: object):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5
    input = Input(input_shape, name='input')
    x = input
    x = Conv1D(64, 8, padding='same', activation="relu", name="Conv1", kernel_initializer="glorot_uniform")(x)
    x = Dropout(dr)(x)
    x = Conv1D(32, 8, padding='same', activation="relu", name="Conv2", kernel_initializer="glorot_uniform")(x)
    x = Dropout(dr)(x)
    x = SumLayer()(x)
    x = BatchNormalization()(x)
    xc = Dense(64, activation='selu', name='dense1')(x)
    xc = BatchNormalization()(xc)

    gaussian_noise_layer = GaussianNoiseLayer(10)
    xc = gaussian_noise_layer(xc)

    x = Reshape((64, 1))(xc)
    x = Bidirectional(LSTM(units=64, return_sequences=True), name='lstm3')(x)
    x = Dropout(dr)(x)
    x = Bidirectional(LSTM(units=64), name='lstm4')(x)
    x = Reshape((1,128))(x)
    multi_head_attention_layer = MultiHeadAttention(num_heads=8, key_dim=128)
    attention_output_1 = multi_head_attention_layer(query=x, key=x, value=x)
    global_avg_pool = GlobalAveragePooling1D()
    x = global_avg_pool(attention_output_1)

    xc = Dense(256, activation='selu', name='dense4')(x)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)

    xd = Dense(classes, activation='softmax', name='xd')(xc)

    model = Model(inputs=input, outputs=xd)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

def PrunedModel(weights = None,
              input_shape=[512, 2],
              classes = 11,
              **kwargs: object):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5
    input = Input(input_shape, name='input')
    x = input
    x = Conv1D(64, 8, padding='same', activation="relu", name="Conv1", kernel_initializer="glorot_uniform")(x)
    x = Dropout(dr)(x)
    x = Conv1D(32, 8, padding='same', activation="relu", name="Conv2", kernel_initializer="glorot_uniform")(x)
    x = Dropout(dr)(x)
    x = SumLayer()(x)
    x = BatchNormalization()(x)
    xc = Dense(64, activation='selu', name='dense1')(x)
    xc = BatchNormalization()(xc)
    gaussian_noise_layer = GaussianNoiseLayer(10)  
    xc = gaussian_noise_layer(xc)
    x = Reshape((64, 1))(xc)
    x = PruneBidirectional(LSTM(units=64, return_sequences=True), name='lstm3')(x)
    x = Dropout(dr)(x)
    x = PruneBidirectional(LSTM(units=64), name='lstm4')(x)
    x = Reshape((1,128))(x)
    multi_head_attention_layer = MultiHeadAttention(num_heads=8, key_dim=128)
    attention_output_1 = multi_head_attention_layer(query=x, key=x, value=x)
    global_avg_pool = GlobalAveragePooling1D()
    x = global_avg_pool(attention_output_1)
    xd = Dense(classes, activation='softmax', name='xd')(x)
    model = Model(inputs=input, outputs=xd)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model
if __name__ == '__main__':
    model = Model(None,input_shape=[512,2],classes=11)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    print('models summary:', model.summary())
