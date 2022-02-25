import keras.callbacks
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Input, Flatten


class SwaModel(tf.keras.Model):
    def __init__(self):
        super(SwaModel, self).__init__()
        self.w_avg = None
        self.w2_avg = None
        self.pre_D = None
        self.n_models = 0
        self.K = 20

    def flatten_weights(self):
        flat = tf.zeros(0)
        for layer in self.weights:
            flat = tf.concat([flat, tf.reshape(layer, -1)], axis=0)
        return flat

    def aggregate_model(self):
        # """Aggregate parameters for SWA/SWAG"""
        cur_w = self.flatten_weights()
        cur_w2 = cur_w ** 2

        if self.w_avg is None:
            self.w_avg = cur_w
            self.w2_avg = cur_w2
        else:
            self.w_avg = (self.w_avg * self.n_models + cur_w) / (self.n_models + 1)
            self.w2_avg = (self.w2_avg * self.n_models + cur_w2) / (self.n_models + 1)

        if self.pre_D is None:
            self.pre_D = tf.identity(cur_w)[:, None]
        else:
            # Record weights, measure discrepancy with average later
            self.pre_D = tf.concat((self.pre_D, cur_w[:, None]), axis=1)
            if self.pre_D.shape[1] > self.K:
                self.pre_D = self.pre_D[:, 1:]

        self.n_models += 1

    def load_weights(self, arr):
        currWeights = self.get_weights()
        currIndex = 0
        newWeights = []
        for i, layer in enumerate(currWeights):
            n = np.prod(layer.shape)
            w = arr[currIndex:currIndex+n]
            currIndex+=n
            newWeights.append(tf.reshape(w, layer.shape))
        self.set_weights(newWeights)

    def sample_weights(self, scale=0.5):
        """Sample weights using SWAG:
        - w ~ N(avg_w, 1/2 * sigma + D . D^T/2(K-1))
            - This can be done with the following matrices:
                - z_1 ~ N(0, I_d); d the number of parameters
                - z_2 ~ N(0, I_K)
            - Then, compute:
            - w = avg_w + (1/sqrt(2)) * sigma^(1/2) . z_1 + D . z_2 / sqrt(2(K-1))
        """
        avg_w = self.w_avg  # [K]
        avg_w2 = self.w2_avg  # [K]
        D = self.pre_D - avg_w[:, None]  # [d, K]
        d = avg_w.shape[0]
        K = self.K
        z_1 = tf.random.normal((1, d))
        z_2 = tf.random.normal((K, 1))

        assert D.shape[1] == K, 'Not enough models aggregated'
        w = avg_w[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 * tf.abs(
            avg_w2 - avg_w ** 2) ** 0.5
        w += scale * tf.transpose(D @ z_2) / np.sqrt(2 * (K - 1))
        w = w[0]

        self.load_weights(w)
        return w

    def forward_swag(self, x, scale=0.5):
        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)
        return self.call(x)


class SimplerLeaky(tf.keras.Model):
    def __init__(self):
        # Initialize the necessary components of tf.keras.Model
        super(SimplerLeaky, self).__init__()
        # Now we initalize the needed layers - order does not matter.
        # -----------------------------------------------------------
        # Flatten Layer
        self.flatten = Flatten()
        # First Dense Layer

        self.dense1 = Dense(128)
        self.act1 = LeakyReLU(alpha=0.1)
        self.dense2 = Dense(128)
        self.act2 = LeakyReLU(alpha=0.1)
        self.dense3 = Dense(128)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return self.dense3(x) # Return results of Output Layer


class SwaLeaky(SwaModel):
    def __init__(self, nOut):
        # Initialize the necessary components of tf.keras.Model
        super(SwaLeaky, self).__init__()
        # Now we initalize the needed layers - order does not matter.
        # -----------------------------------------------------------
        # Flatten Layer
        self.flatten = Flatten()
        # First Dense Layer

        self.dense1 = Dense(128)
        self.act1 = LeakyReLU(alpha=0.1)
        self.dense2 = Dense(128)
        self.act2 = LeakyReLU(alpha=0.1)
        self.dense3 = Dense(nOut)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return self.dense3(x) # Return results of Output Layer


class SwaCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.aggregate_model()



if __name__ == "__main__":
    model = SwaLeaky(1)

    model.compile(loss='mse')
    xs = tf.random.uniform((1000, 2))
    ys = tf.reduce_sum(xs, axis=1, keepdims=True)
    model.fit(xs, ys, epochs=50, callbacks=[SwaCallback()])
    yp = model.predict(xs)
    model.aggregate_model()

    print(model.summary())
    print(model.forward_swag(xs[:2]))
    print(model.forward_swag(xs[:2]))