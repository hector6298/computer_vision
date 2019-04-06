import numpy as np
import tensorflow as tf
import cv2

class detector:
    def __init__(self, model_path ='detect.tflite',path_labels = 'labelmap.txt'):

        self.category_index = self.load_labels(path_labels)
        self.interpreter = tf.lite.Interpreter(model_path = model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def detect(self,image,threshold = 0.6):
        img = cv2.resize(image, (self.input_details[0]['shape'][2],self.input_details[0]['shape'][1]))
        frame = np.asarray(img)
        frame = np.expand_dims(frame, axis=0)
        #frame = (2.0 / 255.0) * frame - 1.0
        #frame = frame.astype('float32')
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        loc = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # Find detected loc coordinates
        return self._boxes_coordinates(image,
                            np.squeeze(loc[0]),
                            np.squeeze(classes+1).astype(np.int32),
                            np.squeeze(scores[0]),
                            min_score_thresh=threshold)

    def close(self):
        pass
    def load_labels(self,filename):
        my_labels = []
        input_file = open(filename, 'r')
        for l in input_file:
            my_labels.append(l.strip())
        return my_labels

    def _boxes_coordinates(self,
                            image,
                            loc,
                            classes,
                            scores,
                            max_boxes_to_draw=10,
                            min_score_thresh=.5):
    
        if not max_boxes_to_draw:
            max_boxes_to_draw = loc.shape[0]
        number_boxes = min(max_boxes_to_draw, loc.shape[0])
        person_boxes = []
        # person_labels = []
        for i in range(number_boxes):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(loc[i].tolist())
                ymin, xmin, ymax, xmax = box

                im_height, im_width, _ = image.shape
                left, right, top, bottom = [int(z) for z in (xmin * im_width, xmax * im_width,
                                                             ymin * im_height, ymax * im_height)]

                person_boxes.append([(left, top), (right, bottom), scores[i], self.category_index[classes[i]]])
        return person_boxes  