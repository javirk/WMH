import functools
import numpy as np
import tensorflow as tf
from time import time
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, LeakyReLU, Flatten, Dense, Conv2DTranspose, Reshape
from datetime import datetime
import utils.nns as nns


class WGANGP:
    def __init__(self, train_ds=None, epochs=10000, LAMBDA=10, BATCH_SIZE=8, checkpoint_dir='', TRAINING_RATIO=8,
                 save_interval=50, output_dir='output/', restore_check=False, apply_fourier=True, nbins=10, log_interval=20,
                 spectral_norm=True, new_nets=True, input_dim=512):
        assert input_dim == 128 or input_dim == 512, 'La dimensión de entrada solo puede ser 128 o 512.'
        self.input_dim = input_dim
        self.LAMBDA = LAMBDA
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAINING_RATIO = TRAINING_RATIO
        self.epochs = epochs
        self.train_ds = train_ds
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.output_dir = output_dir
        self.apply_fourier = apply_fourier
        self.nbins = nbins
        self.fourier_real = None
        self.spectral_norm = spectral_norm
        self.new_nets = new_nets
        self.height_images = 200 if self.new_nets else 256

        self.discriminator, self.generator = self.get_networks()

        self.generator_optimizer, self.discriminator_optimizer = self.optimizers()

        self.checkpoint, self.checkpoint_prefix, self.manager = self.create_checkpoints(checkpoint_dir)
        self.make_dirs()
        self.train_summary_writer = self.writers_tensorboard()

        if restore_check:
            print(
                'The model will be trained for ' + str(self.epochs) + ' epochs and will restore last saved checkpoint')
            try:
                self.checkpoint.restore(self.manager.latest_checkpoint)
                print('Checkpoint restored from ' + self.manager.latest_checkpoint)
            except Exception as e:
                print('Error while restoring a checkpoint')
                print(e)
        else:
            print('Training will not use any saved checkpoints. Apply Fourier = ' + str(apply_fourier))

    def get_networks(self):
        # Discriminator
        if self.spectral_norm:
            disc = nns.Discriminator_SN()
        elif self.new_nets:
            disc = nns.Discriminator_prog()
        else:
            disc = nns.Discriminator_original()

        # Generator
        if self.new_nets:
            if self.input_dim == 512:
                gen = nns.Generator_prog_512()
            else:
                gen = nns.Generator_prog()
        else:
            gen = nns.Generator_original()

        return disc, gen

    @staticmethod
    def optimizers():
        generator_optimizer = tf.keras.optimizers.Adam(0.0001)
        discriminator_optimizer = tf.keras.optimizers.Adam(0.0004)

        return generator_optimizer, discriminator_optimizer

    def discriminator_step(self, x_real):
        with tf.GradientTape() as t:
            z = np.random.normal(0, 1, (self.BATCH_SIZE, self.input_dim)).astype('f')
            x_fake = self.generator(z, training=True)

            x_real_from_discriminator = self.discriminator(x_real, training=True)
            x_fake_from_discriminator = self.discriminator(x_fake, training=True)

            D_loss_fake, D_loss_real = self.discriminator_loss(x_fake_from_discriminator, x_real_from_discriminator)
            gp = self.gradient_loss(x_real, x_fake)
            if self.apply_fourier:
                fl = self.fourier_loss(x_real, x_fake)
            else:
                fl = 0

            D_loss_total = D_loss_fake + D_loss_real + gp * self.LAMBDA + self.LAMBDA * fl

        D_grad = t.gradient(D_loss_total, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(D_grad, self.discriminator.trainable_variables))

        return {'d_loss': D_loss_total, 'gp_loss': gp, 'fourier_loss': fl}

    def fourier_loss(self, x_real, x_fake, axes=(1, 2)):
        fourier_fake = self.get_fourier_spectrum(x_fake, axes=axes)
        fl, _ = self.compute_frechet_distance(self.fourier_real, fourier_fake)

        return fl

    def get_fourier_spectrum(self, dataset, axes=(1, 2)):
        fshift = np.fft.fftshift(np.fft.fft2(dataset, axes=axes))
        spectrum = 20 * np.log(np.abs(fshift))
        curvas = self.radial_Average(spectrum, self.nbins)# These results are not normalized. Should be divided by sqrt(3)

        return {'media': curvas.mean(axis=0), 'std': curvas.std(axis=0)}


    @staticmethod
    def radial_Average_image(image, nbins, point_around=None, corners=True):
        if not point_around:
            x0 = image.shape[0] // 2
            y0 = image.shape[1] // 2
        else:
            x0, y0 = point_around[0], point_around[1]

        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        R = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        if corners:
            d = np.sqrt(max(image.shape[0] - x0, x0) ** 2 + max(image.shape[1] - y0, y0) ** 2)
        else:
            d = max(max(image.shape[0] - x0, x0), max(image.shape[1] - y0, y0))

        blow = np.array([i + nbins / d for i in range(nbins)])
        bhigh = np.array([i + nbins / d for i in range(1, nbins + 1)])

        # calculate the mean
        f = lambda rmin, rmax, i_channel: image[(R >= rmin) & (R < rmax), i_channel].mean()

        mean = []
        for i_channel in range(image.shape[2]):
            mean.append(np.vectorize(f)(blow, bhigh, i_channel))

        mean = np.vstack(mean)

        return mean  # ¿Mejor devolver el interpolado también?

    def radial_Average(self, images, nbins, point_around=None, corners=True):
        medias = []
        for image in images:
            mean = self.radial_Average_image(image, nbins, point_around, corners)
            mean /= np.amax(mean)
            mean = np.linalg.norm(mean, axis=0)
            medias.append(mean)
        medias = np.vstack(medias)

        return medias

    @staticmethod
    def compute_frechet_distance(p, q):
        def _euclidea(v1, v2):
            return np.linalg.norm(v1 - v2)

        def _frechet_distribuciones(mu1, mu2, std1, std2):
            return _euclidea(mu1, mu2) + std1 + std2 - 2 * np.sqrt(std1 * std2)

        def _elemento_frechet(i, j):
            if i == 0 and j == 0:
                return _frechet_distribuciones(p['media'][i], q['media'][j], p['std'][i], q['std'][j])
            elif i > 0 and j == 0:
                return max(c[i - 1, 0], _frechet_distribuciones(p['media'][i], q['media'][0], p['std'][i], q['std'][0]))
            elif i == 0 and j > 0:
                return max(c[0, j - 1], _frechet_distribuciones(p['media'][0],  q['media'][j], p['std'][0], q['std'][j]))
            elif i > 0 and j > 0:
                return max(min(c[i - 1, j], c[i - 1, j - 1], c[i, j - 1]),
                           _frechet_distribuciones(p['media'][i],  q['media'][j], p['std'][i], q['std'][j]))

        c = - np.ones((len(p['media']), len( q['media'])))

        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i, j] = _elemento_frechet(i, j)

        distance = c[-1, -1]

        return distance, c

    @tf.function
    def generator_step(self):
        with tf.GradientTape() as t:
            z = np.random.normal(0, 1, (self.BATCH_SIZE, self.input_dim)).astype('f')
            x_fake = self.generator(z, training=True)
            x_fake_from_discriminator = self.discriminator(x_fake, training=True)
            G_loss = self.generator_loss(x_fake_from_discriminator)

        G_grad = t.gradient(G_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(G_grad, self.generator.trainable_variables))

        return {'g_loss': G_loss}

    def interpolate(self, x_real, x_fake):
        shape = [tf.shape(x_real)[0]] + [1] * (x_real.ndim - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = x_real + alpha * (x_fake - x_real)
        inter.set_shape(x_real.shape)
        return inter

    def gradient_loss(self, x_real, x_fake):
        # Based on https: // github.com / LynnHo / DCGAN - LSGAN - WGAN - GP - DRAGAN - Tensorflow - 2
        x = self.interpolate(x_real, x_fake)

        f = functools.partial(self.discriminator, training=True)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        gp = tf.reduce_mean((tf.norm(grad) - 1) ** 2)

        return gp

    @staticmethod
    def discriminator_loss(x_fake_from_discriminator, x_real_from_discriminator):
        d_loss_fake = tf.reduce_mean(x_fake_from_discriminator)
        d_loss_real = - tf.reduce_mean(x_real_from_discriminator)

        return d_loss_fake, d_loss_real

    @staticmethod
    def generator_loss(x_fake_from_discriminator):
        g_loss = - tf.reduce_mean(x_fake_from_discriminator)
        return g_loss

    def create_checkpoints(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

        return checkpoint, checkpoint_prefix, manager

    def make_dirs(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists('logs/'):
            os.mkdir('logs/')

        if not os.path.exists(self.output_dir + 'FLAIR/'):
            os.mkdir(self.output_dir + 'FLAIR/')

        if not os.path.exists(self.output_dir + 'T1/'):
            os.mkdir(self.output_dir + 'T1/')

        if not os.path.exists(self.output_dir + 'Mask/'):
            os.mkdir(self.output_dir + 'Mask/')

    def plot_channel(self, x, channel, name, n_images_to_present, cols, epoch):
        fig = plt.figure()
        for n, image in enumerate(x):
            a = fig.add_subplot(np.ceil(n_images_to_present / float(cols)), cols, n + 1)
            plt.imshow(image[..., channel], cmap='gray')
        if not os.path.isdir(self.output_dir + name + '/'):
            os.mkdir(self.output_dir + name + '/')

        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images_to_present)
        fig.savefig(self.output_dir + name + '/epoch_' + str(epoch) + '.png')
        plt.close(fig)

    def generate_images(self, n_images_to_present, epoch):
        cols = 2
        z = np.random.normal(0, 1, (n_images_to_present, 128)).astype('f')

        x = self.generator(z)

        self.plot_channel(x, 1, 'FLAIR', n_images_to_present, cols, epoch)
        self.plot_channel(x, 0, 'T1', n_images_to_present, cols, epoch)
        self.plot_channel(x, 2, 'Mask', n_images_to_present, cols, epoch)

    def generate_images_tb(self, n_images_to_present, epoch):
        z = np.random.normal(0, 1, (n_images_to_present, self.input_dim)).astype('f')
        x = self.generator(z)

        t1 = np.reshape(x[..., 0] + 1 / 2, (-1, x.shape[1], x.shape[2], 1))
        flair = np.reshape(x[..., 1] + 1 / 2, (-1, x.shape[1], x.shape[2], 1))
        mask = np.reshape(x[..., 2] + 1 / 2, (-1, x.shape[1], x.shape[2], 1))

        with self.train_summary_writer.as_default():
            tf.summary.image('FLAIR', flair, max_outputs=n_images_to_present, step=epoch)
            tf.summary.image('T1', t1, max_outputs=n_images_to_present, step=epoch)
            tf.summary.image('Mask', mask, max_outputs=n_images_to_present, step=epoch)

    def fit(self, dataset, initial_epoch=0):
        assert dataset.shape[1] == dataset.shape[2] and dataset.shape[1] == self.height_images, \
            'The size of the dataset must be batches x' + str(self.height_images) + 'x' + str(self.height_images) + 'x3'

        if self.apply_fourier:
            self.fourier_real = self.get_fourier_spectrum(dataset)

        n_images = dataset.shape[0]
        n_minibatches = int(n_images // (self.BATCH_SIZE * self.TRAINING_RATIO))
        training_ratio_epoch = initial_epoch * n_minibatches * self.TRAINING_RATIO

        for epoch in range(self.epochs):
            start = time()
            np.random.shuffle(dataset)
            minibatches_size = self.BATCH_SIZE * self.TRAINING_RATIO

            for i in range(int(n_images // (self.BATCH_SIZE * self.TRAINING_RATIO))):
                discriminator_minibatches = dataset[i * minibatches_size: (i + 1) * minibatches_size]

                for j in range(self.TRAINING_RATIO):
                    image_batch = discriminator_minibatches[j * self.BATCH_SIZE: (j + 1) * self.BATCH_SIZE]
                    disc_loss = self.discriminator_step(image_batch)
                    training_ratio_epoch += 1

                gen_loss = self.generator_step()

            print('Epoch ' + str(epoch) + ' took ' + str(time() - start))

            if (epoch + 1) % self.log_interval == 0:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Generator loss', gen_loss['g_loss'], step=epoch + 1)
                    tf.summary.scalar('Discriminator loss', disc_loss['d_loss'], step=training_ratio_epoch)
                    tf.summary.scalar('Gradient Penalty loss', disc_loss['gp_loss'], step=training_ratio_epoch)
                    tf.summary.scalar('Fourier loss', disc_loss['fourier_loss'],  step=training_ratio_epoch)

            if (epoch + 1) % self.save_interval == 0:
                # self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                self.manager.save()
                print('Checkpoint saved.')
                self.generate_images_tb(3, epoch + 1)
                # self.generate_images(10, epoch + 1)
                print('Images saved.')

    @staticmethod
    def writers_tensorboard():
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        return train_summary_writer