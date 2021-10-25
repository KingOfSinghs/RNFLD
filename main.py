from load_data import load_data
from generator import load_generators
from train import train
from test import test
from model import RNFLModel

import time
from importlib import reload
import tensorflow as tf
import tensorflow.keras.backend as k
k.clear_session()

import config

# tensorboard --logdir=logs/fit

def main():
    print("[INFO] Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()

    weights_dir = config.WEIGHTS_DIR # one weight dir

    # multi run with decreasing LRs
    for i, lr in config.LRS:
        data = load_data(config)

        generators = load_generators(data, config)

        model = RNFLModel(i, lr=lr, weights=weights_dir, config=config)
        model.compile()

        train(model, generators, config)
        test(model, data, config)

        print(f'\nINFO] MODEL {config.MODEL} RUN {i} COMPLETE')

        reload(config) # reload timestamps
        time.sleep(5) # gpu cool down

        print('♕'*40)
        print('♕'*40)


    print('\n[INFO] PROGRAM FINISHED')
    exit()


if __name__ == '__main__':
    main()