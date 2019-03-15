import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
from densenet161 import DenseNet
from keras.optimizers import SGD

class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model
        model = DenseNet(reduction=0.5, classes=1000, weights_path="model/model.h5")
        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model._make_predict_function()
        self._model = model

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference with caffe2
        results =  self._model.predict(inputAsNpArr)
        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output
