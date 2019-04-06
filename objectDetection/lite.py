import tensorflow as tf
import cv2
import numpy as np

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()
input = interpreter.get_input_details()
output = interpreter.get_output_details()
labels = load_labels('labelmap.txt')

cap = cv2.VideoCapture(0) 

while 1:
  ret, frame = cap.read()
  if frame is not None:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input[0]['shape'][2],input[0]['shape'][1]))
    frame = np.asarray(img)
    frame = np.expand_dims(frame,axis=0)
    
    interpreter.set_tensor(input[0]["index"], frame)
    interpreter.invoke()
    
  
    output_data = interpreter.get_tensor(output[1]["index"])
    score = interpreter.get_tensor(output[2]['index'])
    results = np.squeeze(output_data)
    score = np.squeeze(score+1).astype(int)
    results = np.delete(results,np.where(score < 0.6))
    print(results)
    for i in range(len(results)):
      print(str(int(results[i]))+":"+ labels[int(results[i])])
    
   
