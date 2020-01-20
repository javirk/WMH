from utils.model import WGANGP
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

datasets_path = '../00. Datasets/'
BATCH_SIZE = 8
TRAINING_RATIO = 1
IMAGES_PER_EPOCH = 200

modelo = WGANGP(epochs=10000, BATCH_SIZE=BATCH_SIZE, checkpoint_dir='checkpoints/', log_interval=2, spectral_norm=False,
                save_interval=50, TRAINING_RATIO=TRAINING_RATIO, tipo_latente='normal', apply_fourier=False, plot_weights=True)

ds = np.load(datasets_path + 'total_three_datasets_sorted.npy')

modelo.fit(dataset=ds, images_per_epoch=IMAGES_PER_EPOCH)
