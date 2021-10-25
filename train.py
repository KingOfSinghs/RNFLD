import numpy as np
import tensorflow as tf

def train(model, generators, config):
    train_gen, val_gen = generators

    # set up tensorboard images
    x, y = val_gen.load_batch(0)
    file_writer = tf.summary.create_file_writer(config.LOG_DIR)
    with file_writer.as_default():
        images = np.reshape(x[:10], (-1, 256, 256, config.CHANNELS))
        tf.summary.image("10 training data examples", images, max_outputs=10, step=0)
    file_writer.close()


    model.fit(train_gen,
          validation_data = val_gen,
          val_steps=len(val_gen),
          epochs = config.EPOCHS,
          steps_per_epoch=len(train_gen) * config.STEPS_FACTOR,
          callbacks=config.CALLBACKS)
