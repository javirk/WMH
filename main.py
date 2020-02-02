from utils.model import WGANGP
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

datasets_path = '../00. Datasets/'
BATCH_SIZE = 6
TRAINING_RATIO = 2
IMAGES_PER_EPOCH = 200
LAMBDA = 0

modelo = WGANGP(epochs=10000, BATCH_SIZE=BATCH_SIZE, LAMBDA=LAMBDA, checkpoint_dir='checkpoints/', log_interval=2, spectral_norm=False,
                save_interval=50, TRAINING_RATIO=TRAINING_RATIO, tipo_latente='uniforme', apply_fourier=False, plot_weights=True)

ds = np.load(datasets_path + 'muestra_seleccionada_256_sin_ceros.npy')

modelo.fit(dataset=ds)
