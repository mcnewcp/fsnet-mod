import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Layer, Input, Dense, Dropout, LeakyReLU
from keras.initializers import Constant
from keras.optimizers import RMSprop
import numpy as np
import math


class tinyLayerE(Layer):
    def __init__(
        self,
        output_dim,
        u,
        bins,
        start_temp=10.0,
        min_temp=0.1,
        alpha=0.99999,
        **kwargs
    ):
        self.output_dim = output_dim
        self.u = K.constant(u)
        self.bins = bins
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(tinyLayerE, self).__init__(**kwargs)

    def build(self, input_shape):
        self.temp = self.add_weight(
            name="temp",
            shape=[],
            initializer=Constant(self.start_temp),
            trainable=False,
        )
        self.tinyW = self.add_weight(
            name="tinyW",
            shape=(self.bins, self.output_dim),
            initializer="uniform",
            trainable=True,
        )
        super(tinyLayerE, self).build(input_shape)

    def call(self, X, training=None):
        al = K.softmax(K.dot(self.u, self.tinyW))
        al = K.transpose(al)
        logits = K.log(10 * K.maximum(K.minimum(al, 0.9999999), K.epsilon()))
        uniform = K.random_uniform(logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (logits + gumbel) / temp
        samples = K.softmax(noisy_logits)
        discrete_logits = K.one_hot(K.argmax(logits), logits.shape[1])
        self.logits = samples
        dl = np.zeros(self.logits.shape)
        p = K.get_value(self.logits)
        for i in range(dl.shape[0]):
            ind = np.argmax(p, axis=None)
            x = ind // dl.shape[1]
            y = ind % dl.shape[1]
            dl[x][y] = 1
            p[x] = -np.ones(dl.shape[1])
            p[:, y] = -np.ones(dl.shape[0])
            discrete_logits = K.one_hot(K.argmax(K.variable(dl)), dl.shape[1])
        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))
        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class tinyLayerD(Layer):
    def __init__(self, output_dim, u, bins, **kwargs):
        self.output_dim = output_dim
        self.u = K.constant(u)
        self.bins = bins
        super(tinyLayerD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.tinyW = self.add_weight(
            name="tinyW",
            shape=(self.bins, input_shape[1]),
            initializer="uniform",
            trainable=True,
        )
        super(tinyLayerD, self).build(input_shape)

    def call(self, x):
        weights = K.transpose(K.tanh(K.dot(self.u, self.tinyW)))
        return K.dot(x, weights)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def get_utrain(X_train, bins=10):
    u_train = np.zeros([X_train.shape[1], bins], dtype=float)
    for i in range(0, X_train.shape[1]):
        hist = np.histogram(X_train[:, i], bins)
        for j in range(0, bins):
            u_train[i, j] = hist[0][j] * 0.5 * (hist[1][j] + hist[1][j + 1])
    return u_train


def get_alpha(X_train, batch_size=8, min_temp=0.01, start_temp=10.0, num_epochs=100):
    steps_per_epoch = (len(X_train) + batch_size - 1) // batch_size
    alpha = math.exp(math.log(min_temp / start_temp) / (num_epochs * steps_per_epoch))
    return alpha

