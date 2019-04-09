import cv2
import tensorflow as tf
import numpy as np
from classifier import classifier
import argparse

#argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default="img/1.jpg", \
                    help="image to be classified")
parser.add_argument("-m", "--model_file", default="mobilenet_v1_1.0_224_quant.tflite", \
                    help="tensorflow lite model file to be loaded from directory")
parser.add_argument("-l", "--label_file", default="labels_mobilenet_quant_v1_224.txt", \
                    help="Label file")
args = parser.parse_args()

#classifier object initialization and prediction
classifier = classifier(model_path= args.model_file,path_labels= args.label_file)
img =  cv2.imread(args.image,-1)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
if(image is None):
    print('Image could not be loaded. is the directory correct?')
else:
    result,arr = classifier.predict(image)

    #output result
    cv2.putText(img, "class: "+result, (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)
    print("class: "+ result)
    cv2.imshow("classification", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()