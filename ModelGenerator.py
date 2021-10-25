from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class ModelGenerator:
    def __init__(self, config):
        self.img_dim = config.IMG_DIM
        self.classes = config.CLASSES

        self.activation = 'softmax' if self.classes > 1 else 'sigmoid'
        self.input_shape = config.IMG_SHAPE

        # custom models
        self.models_ = dict(
            VGG19 = VGG19(include_top=False,
                          input_shape=self.input_shape,
                          classes=self.classes, weights=None),

            ResNet50 = ResNet50(include_top=False,
                         input_shape=self.input_shape,
                         classes=self.classes, weights=None),

            InceptionV3 = InceptionV3(include_top=False,
                            input_shape=self.input_shape,
                            classes=self.classes, weights=None),

            Xception = Xception(include_top=False,
                         input_shape=self.input_shape,
                         classes=self.classes, weights=None),

            EfficientNetB0 = EfficientNetB0(include_top=False,
                               input_shape=self.input_shape,
                               classes=self.classes, weights=None),
        )

    def get(self, name):
        try:
            model = self.models_[name]
            model = self.build_model(model, name)
            return model
        except KeyError:
            print(f'[ERROR] Check model spelling: {name}')
            exit()


    def build_model(self, backbone, name):
        backbone_out = backbone.output
        gap = GlobalAveragePooling2D(name='pooling_layer')(backbone_out)
        output = Dense(units=self.classes, activation=self.activation, name='output_layer')(gap)
        model = Model(inputs=backbone.input, outputs=output, name=name)
        return model