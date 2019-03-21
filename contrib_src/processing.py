from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json
import cv2

class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            image = cv2.resize(np.array(image).astype(np.float32), (224, 224)).astype(np.float32)
            image = image[:, :, ::-1].copy()
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return image

    def _preprocessAfterConversionToNumpy(self, npArr):
        if npArr.shape[1] > 3:
            npArr = npArr[0:3,:,:]
        elif npArr.shape[1] < 3:
            npArr = npArr[[0],:,:]
            npArr = np.concatenate((npArr, npArr[[0],:,:]), axis = 1)
            npArr = np.concatenate((npArr, npArr[[0],:,:]), axis = 1)
        npArr[:,:,0] = (npArr[:,:,0] - 103.94) * 0.017
        npArr[:,:,1] = (npArr[:,:,1] - 116.78) * 0.017
        npArr[:,:,2] = (npArr[:,:,2] - 123.68) * 0.017
        npArr = np.expand_dims(npArr, axis=0)
        return npArr

    def computeOutput(self, inferenceResults):
        probs = np.squeeze(np.asarray(inferenceResults))
        with open("model/labels.json") as jsonFile:
            labels = json.load(jsonFile)
        result = []
        for i in range (len(probs)):
            obj = {'label': str(labels[str(i)]),
                    'probability': float(probs[i])}
            result.append(obj)
        return result
