from ModelGenerator import ModelGenerator
from tensorflow.keras.optimizers import Adam

class RNFLModel:
    def __init__(self, run, lr, weights, config):
        self.name = config.MODEL
        self.loss = config.LOSS
        self.metrics = config.METRICS
        self.save = config.SAVE_WEIGHTS

        self.lr = lr
        self.run = run
        self.weights = weights
        self.optimizer = Adam(lr=lr)

        # retrieve model from model generator
        model_gen = ModelGenerator(config)
        self.model = model_gen.get(self.name)

    def get_model(self):
        return self.model

    def predict(self, x):
        return self.model.predict(x)

    def save_weights(self, fp):
        self.model.save_weights(fp, overwrite=True)

    def compile(self, display_summary=False):
        if display_summary:
            print(f'[INFO] Compiling {self.name}...')
            print(26*'♕')
            self.model.summary()

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.load_weights(self.weights)

        return self.model

    def fit(self, generator, validation_data, epochs, steps_per_epoch, callbacks, val_steps):
        print(f'[INFO] TRAINING MODEL {self.name}...')

        self.model.fit(
              generator,
              epochs=epochs,
              validation_data=validation_data,
              validation_steps=val_steps,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks, verbose=1
        )

        if self.save:
            self.save_weights(self.weights)
            print('[INFO] WEIGHTS HAVE BEEN SAVED')

    def load_weights(self, fp):
        if self.run == 1:
            return # run 1 - no weights
        else:
            self.model.load_weights(fp)
            print('[INFO] Weights loaded')