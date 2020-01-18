from utils.dataset import train_pipeline
from utils.model import WGANGP
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

datasets_path = '../00. Datasets/'
BATCH_SIZE = 8

modelo = WGANGP(epochs=10000, BATCH_SIZE=BATCH_SIZE, checkpoint_dir='checkpoints/', log_interval=2, spectral_norm=False, save_interval=50)

ds = np.load(datasets_path + 'muestra_seleccionada_200_sin_ceros.npy')

modelo.fit(dataset=ds)
