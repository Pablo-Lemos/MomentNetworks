from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Input


class SimplerLeaky(tf.keras.Model):
    def __init__(self):
        # Initialize the necessary components of tf.keras.Model
        super(SimplerLeaky, self).__init__()
        # Now we initalize the needed layers - order does not matter.
        # -----------------------------------------------------------
        # Flatten Layer
        self.flatten = keras.layers.Flatten()
        # First Dense Layer
        self.dense1 = keras.layers.Dense(128, activation = tf.nn.relu)
        # Output Layer
        self.dense2 = keras.layers.Dense(10)

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