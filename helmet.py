import PySimpleGUI as sg
import cv2
import numpy as np
import os

confThreshold = 0.5
nmsThreshold = 0.5 
font = cv2.FONT_HERSHEY_PLAIN
classes = []
with open(r"./HelmetDetection/Helmet.names", "r") as f:
    classes = [line.strip() for line in f.readlines ()]
    
net = cv2.dnn.readNet ('./HelmetDetection/Helmet.weights' , './HelmetDetection/Helmet.cfg') 
# net - cv2.dnn.readNet ('weight_cfg_dir/yolov3_tiny_custom.weights', 'weight_cfg_dir/yolov3_tiny_custom.cfg')
m=0
n=0 
sg.theme('DarkAmber') # please make your creations colorful

layout = [ [sg.Text('Filename')],
           [sg. Input(key='file'), sg.FileBrowse()],
           [sg.OK(), sg.Cancel()]]

window = sg.Window ('Get filename example', layout)


event, values = window.read() 
file=values['file'] 
#window.close()

#for filename in os.listdir (path):
img = cv2.imread(file) 
img = cv2.resize(img, (900, 600))

height, width, _ =img.shape 
blob = cv2.dnn.blobFromImage (img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False) 
net.setInput(blob)


output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = [] 
confidences = [] 
class_ids = []
for output in layerOutputs: 
    for detection in output:
        scores = detection [5:] 
        id = np.argmax(scores) 
        confidence = scores[id] 
        if confidence > confThreshold:
            w, h = (int (detection [2] * width), int (detection [3] * height))
            x, y = (int ((detection[0] * width) -w/2), int((detection [1] * height) -h / 2))

            
            boxes.append([x, y, w, h]) 
            confidences.append(float(confidence))
            class_ids.append(id)

indexes = cv2.dnn.NMSBoxes (boxes, confidences, confThreshold ,nmsThreshold)

if len (indexes) > 0: 
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str (classes [class_ids[i]]) 
        confidence = str((int (confidences [i] * 100)))

        
        if label == 'WithHelmet':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText (img, label + " " + confidence + '%', (x, y - 8), font, 2, (0, 255, 0), 2) 
            m=m+1


        if label == 'NoHelmet':
            cv2 .rectangle (img, (x, y), (x + w, y + h), (0,0,255), 2) 
            cv2 .putText (img, label + " " + confidence + '%', (x, y - 18), font, 2,(0,0,255) , 2)
            n=n+1

#print(m)
#print(n)
img = cv2.putText (img, 'WithHelmet :', (20,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
img = cv2.putText (img, str(m), (280,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
img - cv2.putText (img, 'NoHelmet :', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 
img = cv2.putText (img, str(n), (280,60), cv2. FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
cv2.imshow ('Mask Detection Yolo-V4', img) 
cv2.waitKey(0) 
window.close()
