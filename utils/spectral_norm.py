import tensorflow as tf
from tensorflow.keras.layers import Wrapper


class SpectralNorm(Wrapper):
    # From https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/PRjyj6tiQvU
    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNorm, self).__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('Invalid layer for SpectralNorm.')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_variable(shape=(1, self.w_shape[-1]), initializer=tf.random_normal_initializer(),
                                       name='sn_u', trainable=False, dtype=tf.float32) # Please use layer.add_weight

        super(SpectralNorm, self).build()

    @tf.function
    def call(self, inputs, training=None):
        self._compute_weights(training)
        output = self.layer(inputs)

        return output

    def _compute_weights(self, training):
        iteration = self.iteration
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = tf.identity(self.u)
        v_hat = None

        for _ in range(self.iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = tf.nn.l2_normalize(u_)

        if training == True: self.u.assign(u_hat)
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))

        w_norm = self.w / sigma

        self.layer.kernel = w_norm

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())