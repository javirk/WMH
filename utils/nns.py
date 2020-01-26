from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, LeakyReLU, Flatten, Dense, Conv2DTranspose, Reshape, \
    AveragePooling2D, UpSampling2D, Input
from utils.spectral_norm import SpectralNorm
from tensorflow.keras.backend import permute_dimensions
import tensorflow as tf
import numpy as np


def Generator_original():
    """Creates a generator model that takes a 128-dimensional noise vector as a "seed",
        and outputs images of size 256x256x3."""
    model = Sequential()

    model.add(Dense(4 * 4 * 2048, input_dim=128))
    # model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((4, 4, 2048), input_shape=(4 * 4 * 2048,)))
    bn_axis = -1

    model.add(Conv2DTranspose(1024, 4, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, 4, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, 4, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, 4, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, 4, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh'))
    # El output de esta última es 256x256x3

    return model


def Discriminator_original():
    model = Sequential()
    model.add(Convolution2D(32, 5, padding='same', strides=[2, 2], input_shape=(256, 256, 3)))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, 5, kernel_initializer='random_normal', strides=[2, 2], padding='same'))
    model.add(LeakyReLU())

    model.add(Convolution2D(128, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Convolution2D(256, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Convolution2D(512, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Convolution2D(1024, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Flatten())
    # model.add(Dense(1024 * 4 * 4, kernel_initializer='random_normal'))
    # model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='random_normal'))

    return model


def Discriminator_prog():
    model = Sequential()
    model.add(Convolution2D(16, 1, padding='same', kernel_initializer='random_normal', input_shape=(200, 200, 3), activation=LeakyReLU()))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(16, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(32, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(32, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(64, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(64, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(128, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(128, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(256, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(256, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(512, 4, padding='same', strides=[4, 4], kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(Dense(1, kernel_initializer='random_normal'))

    return model

def pixel_wise_normalization(epsilon=1e-8):
    return tf.keras.layers.Lambda(lambda y: y * tf.math.rsqrt(tf.reduce_mean(tf.square(y), axis=1, keepdims=True) + epsilon))


def resize_images(x, height_factor, width_factor, data_format,
                  interpolation='nearest'):
    """Resizes the images contained in a 4D tensor -- Overriden from keras Backend.

    Arguments:
      x: Tensor or variable to resize.
      height_factor:
      width_factor:
      data_format: One of `"channels_first"`, `"channels_last"`.
      interpolation: A string, one of `nearest` or `bilinear`.

    Returns:
      A tensor.

    Raises:
      ValueError: in case of incorrect value for
        `data_format` or `interpolation`.
    """
    if data_format == 'channels_first':
        rows, cols = 2, 3
    elif data_format == 'channels_last':
        rows, cols = 1, 2
    else:
        raise ValueError('Invalid `data_format` argument: %s' % (data_format,))

    original_shape = tuple(x.shape.as_list())
    new_shape = original_shape[rows:cols + 1]
    new_shape *= tf.constant(np.array([height_factor, width_factor], dtype='float32'))
    new_shape = tf.cast(new_shape, dtype=tf.int32)

    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
    if interpolation == 'nearest':
        x = tf.image.resize(x, new_shape, method='nearest')
    elif interpolation == 'bilinear':
        x = tf.image.resize(x, new_shape, method='bilinear')
    else:
        raise ValueError('interpolation should be one of "nearest" or "bilinear".')

    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 3, 1, 2])

    if original_shape[rows] is None:
        new_height = None
    else:
        new_height = original_shape[rows] * height_factor

    if original_shape[cols] is None:
        new_width = None
    else:
        new_width = original_shape[cols] * width_factor

    if data_format == 'channels_first':
        output_shape = (None, None, new_height, new_width)
    else:
        output_shape = (None, new_height, new_width, None)
    x.set_shape(output_shape)
    return x

class UpSamplingFloat(UpSampling2D):
    def call(self, inputs):
        return resize_images(inputs, self.size[0], self.size[1], self.data_format, interpolation=self.interpolation)


def Generator_prog(use_pixel_norm=True):
    model = Sequential()
    model.add(Input(shape=128))
    model.add(Dense(4 * 4 * 128, input_dim=128))
    model.add(Reshape((4, 4, 128), input_shape=(4 * 4 * 128,)))
    model.add(Convolution2D(128, 4, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(128, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(128, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(128, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(128, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(64, 3, padding='same', activation=LeakyReLU(0.2)))  # 64x64x64
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(32, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(32, 3, padding='same', activation=LeakyReLU(0.2))) # 32x128x128
    model.add(pixel_wise_normalization())

    model.add(UpSamplingFloat(size=(200 / 128, 200 / 128)))
    model.add(Convolution2D(16, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D( 16, 3, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(3, 1, padding='same', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    return model

def Generator_prog_512():
    model = Sequential()

    model.add(Dense(4 * 4 * 512, input_dim=512))
    model.add(Reshape((4, 4, 512), input_shape=(4 * 4 * 512,)))
    model.add(Convolution2D(512, 4, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(512, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(256, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))  # 256x64x64
    model.add(pixel_wise_normalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(128, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))  # 128x128x128
    model.add(pixel_wise_normalization())

    model.add(UpSamplingFloat(size=(200 / 128, 200 / 128)))
    model.add(Convolution2D(16, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())
    model.add(Convolution2D(16, 3, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))   # 16x256x256
    model.add(pixel_wise_normalization())

    model.add(Convolution2D(3, 1, padding='same', kernel_initializer='random_normal', activation=LeakyReLU(0.2)))
    model.add(pixel_wise_normalization())

    return model


def Discriminator_SN():
    model = Sequential()
    model.add(SpectralNorm(Convolution2D(32, 5, padding='same', strides=[2, 2], input_shape=(256, 256, 3))))
    model.add(LeakyReLU())

    model.add(SpectralNorm(Convolution2D(64, 5, kernel_initializer='random_normal', strides=[2, 2], padding='same')))
    model.add(LeakyReLU())

    model.add(SpectralNorm(Convolution2D(128, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2])))
    model.add(LeakyReLU())

    model.add(SpectralNorm(Convolution2D(256, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2])))
    model.add(LeakyReLU())

    model.add(SpectralNorm(Convolution2D(512, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2])))
    model.add(LeakyReLU())

    model.add(SpectralNorm(Convolution2D(1024, 5, kernel_initializer='random_normal', padding='same', strides=[2, 2])))
    model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='random_normal'))

    return model

if __name__ == '__main__':
    import numpy as np
    m = Generator_prog()
    z = np.random.normal(0, 1, (1, 128)).astype('f')

    a = m(z)