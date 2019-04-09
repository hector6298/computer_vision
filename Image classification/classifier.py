import numpy as np
import tensorflow as tf
import cv2

class classifier:
    def __init__(self, model_path ='mobilenet_v1_1.0_224_quant.tflite',path_labels = 'labels_mobilenet_quant_v1_224.txt'):

        self.category_index = self.load_labels(path_labels)
        self.interpreter = tf.lite.Interpreter(model_path = model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.softmax_arr = np.array([])

    def load_labels(self,filename):
        my_labels = []
        input_file = open(filename, 'r')
        for l in input_file:
            my_labels.append(l.strip())
        return my_labels
    
    def predict(self,image):
        img = cv2.resize(image, (self.input_details[0]['shape'][2],self.input_details[0]['shape'][1]))
        frame = np.asarray(img)
        frame = np.expand_dims(frame, axis=0)
        print(frame.shape)
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()
        self.softmax_arr = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
        print(self.softmax_arr)
        out_class = int(np.argmax(self.softmax_arr))
        return self.category_index[out_class],self.softmax_arr
    
    def close(self):
        pass