import os
import tensorflow as tf
import cv2
import time
import numpy as np
from PIL import Image
print("[INFO] Loaded Dependencies")

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
print("[INFO] Loading Model")
PATH_TO_SAVED_MODEL = 'app\\BehavMonitor\\my_ssd_mobilenet_v2_fpnlite_320x320\\saved_model'
net = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print("[INFO] Loaded")

# Create Labels
LABELS = {'1':'Running', '2':'_'}

# Set lowe confidence limit for each detection
confidence = 0.7

def prepare_image(img):
    # convert the color from BGR to RGB then convert to PIL array
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    # resize the array (image) then PIL image
    im_resized = im_pil.resize((320, 320))
    image_expanded = np.expand_dims(im_pil, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_expanded)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    #input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor

# resize OPENCV window size
cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
# initialize camera stream
print("[INFO] starting video stream...")
print("[INFO] Press Esc to close")
vid=cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
    # read frame from camera
    _ , frame = vid.read()

    # grab the frame dimensions
    y, x,_ = frame.shape

    input = prepare_image(frame)

    start = time.time()
    detections = net(input)
    end = time.time()
    fps = 1 / (end-start)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    for i in range(num_detections):
        conf = detections['detection_scores'][i]
        if conf < confidence:
            break
        if detections['detection_classes'][i] == 2:
            continue
        loc = detections['detection_boxes'][i]
        coor = np.squeeze(np.multiply(loc, [y,x,y,x,])).astype(np.int32)
        pred_class = LABELS[str(detections['detection_classes'][i])]
        #print(pred_class)
        #text = "{}  :  {:.2f}%".format(pred_class, conf * 100)
        text = "{}".format(pred_class)
        putY = coor[0] - 10 if coor[0] - 10 > 10 else coor[0] + 10
        cv2.rectangle(frame, (coor[1],coor[0]), (coor[3],coor[2]), (0, 200, 200), 2)
        cv2.putText(frame, text, (coor[1], putY),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,200), 2)
    
    cv2.putText(frame, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # show the output frame
    cv2.imshow("Cam", frame)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# After the loop release the
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

print("[INFO] Closing Process...")