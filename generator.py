from tensorflow.keras.utils import Sequence
from image_operations import resample_img
from augmentation import batchAugmented

import tensorflow as tf
import numpy as np
import math

def load_generators(data, config):
    train_gen = DataGenerator(data['x_train'], data['y_train'], config=config, do_augment=config.AUGMENT)
    val_gen = DataGenerator(data['x_val'], data['y_val'], config=config, do_augment=False)

    return train_gen, val_gen

class DataGenerator(Sequence):

    def __init__(self, x_set, y_set, config, do_augment): # do_augment is false, cannot use config.aug
        self.x = x_set
        self.y = y_set
        self.do_augment = do_augment
        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.img_dim = config.IMG_DIM

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        return self.load_batch(idx)

    def load_batch(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = np.array([resample_img(path, self.img_dim) for path in batch_x])

        # augment
        if self.do_augment:
            batch_x = np.array([batchAugmented(x, self.do_augment) for x in batch_x])

        # augmented images to tensorboard
        if idx == 0:
            file_writer = tf.summary.create_file_writer(self.config.LOG_DIR)
            with file_writer.as_default():
                images = np.reshape(batch_x[0:10], (-1, 256, 256, self.config.CHANNELS))
                tf.summary.image("10 augmented data examples", images, max_outputs=10, step=0)
            file_writer.close()

        return batch_x / 255.0 , batch_y / 1.0