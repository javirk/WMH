from utils.model import WGANGP
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

datasets_path = '../00. Datasets/'
BATCH_SIZE = 6
TRAINING_RATIO = 8
IMAGES_PER_EPOCH = 200
LAMBDA = 10

modelo = WGANGP(epochs=10000, BATCH_SIZE=BATCH_SIZE, LAMBDA=LAMBDA, checkpoint_dir='checkpoints/', log_interval=2, spectral_norm=False,
                save_interval=50, TRAINING_RATIO=TRAINING_RATIO, tipo_latente='uniforme', apply_fourier=False, plot_weights=True, new_nets=False)

ds = np.load(datasets_path + 'borrado-renormalizado.npy')

modelo.fit(dataset=ds, images_per_epoch=IMAGES_PER_EPOCH)
