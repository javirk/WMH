import tensorflow as tf
import numpy as np

def train_pipeline(PATH_FILE, BUFFER_SIZE, n, BATCH_SIZE):
    data = tf.cast(np.load(PATH_FILE), tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(data)
    train_dataset = train_dataset.take(n)
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset