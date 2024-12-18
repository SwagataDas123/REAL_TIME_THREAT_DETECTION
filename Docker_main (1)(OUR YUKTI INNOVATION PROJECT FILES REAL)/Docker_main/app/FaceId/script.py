# import the necessary libraries

print("[INFO] Loading Dependencies...")
import numpy as np
import imutils
import cv2
import os
import time
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras_facenet import FaceNet

print("[INFO] Loading Embedder...")
embedder = FaceNet()
target = [0]*2
#print(os.listdir('app\FaceId\Target\Ahmed_Kaif_Member'))
print("[INFO] Preparing Target Encodings...")
enc =0
for img in os.listdir('app\FaceId\Target\Ahmed_Kaif_Member'):
    enc +=np.array((embedder.extract('app\FaceId\Target\Ahmed_Kaif_Member/'+img))[0]['embedding'])
target[0] = enc/len(os.listdir('app\FaceId\Target\Ahmed_Kaif_Member'))

enc =0
for img in os.listdir('app\FaceId\Target\Swagata_Das_Member'):
    enc +=np.array((embedder.extract('app\FaceId\Target\Swagata_Das_Member/'+img))[0]['embedding'])
target[1] = enc/len(os.listdir('app\FaceId\Target\Swagata_Das_Member'))

print("[INFO] Target Encodings Ready")

# set values for prototxt and .caffemodel which will be used to set up the model
prototxt = 'app\\FaceId\openCV_DNN_Custom\\deploy.prototxt.txt'
model = 'app\\FaceId\\openCV_DNN_Custom\\res10_300x300_ssd_iter_140000.caffemodel'

# Set confidence lvl for the face detection
conf = 0.5

# load Face Detector SSD model architecture and weights
extractor = cv2.dnn.readNetFromCaffe(prototxt, model)

# Target labels
LABELS = ["Ahmed_Kaif","Swagata_Das"]

# resize OPENCV window size
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# initialize camera stream
print("[INFO] starting video stream...")
print("[INFO] Press Esc to Exit...")
vid=cv2.VideoCapture(0)
# loop over the frames from the video stream
while True:
    # read frame from camera and resize to 400 pixels
    ret,frame = vid.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    start = time.time()
    # pass the blob through the network and obtain the detections and
    # predictions
    extractor.setInput(blob)
    detections = extractor.forward()

    end = time.time()
    # calculate the FPS for current frame detection
    fps = (end-start)
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out detections by confidence
        if confidence < conf:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
        # Preparing Face for classification
        face = frame[startY:endY, startX:endX]
        
        # Performing Classification and measuring time for inference
        start = time.time()
        try:
            embedding = np.array(embedder.embeddings([face]))
        except:
            continue
        dis = [np.linalg.norm(embedding - i) for i in target]
        ind = np.argmin(dis, axis=0)
        #set_of_dis.append(dis)
        end = time.time()
        fps += (end-start)
        
        # ind = np.argmax(pred, axis=1)[0]   # Index of identified label
        # pred_conf = pred.max(axis=1)[0]    # Confidence in class
        color = (0,255,0)
        pred_class = 'Unknown'
        #print(ind)
        if min(dis) <= 0.8 :
            color = (0,0,255)
            pred_class = LABELS[ind]
        # draw the bounding box of the face along with the associated
        # probability
        text = "{}  :  {:.2f}".format(pred_class, dis[ind])   #, pred_conf * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            color, 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    fps = 1 / fps
    cv2.putText(frame, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# After the loop release the
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

print("[INFO] Closing Process...")