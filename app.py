from flask import Flask
from flask import Flask, render_template, request,Response
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.preprocessing.image as imag
from tensorflow.python.keras.models import load_model
config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5)


model=load_model('inception_v3/model.h5')

app = Flask(__name__)
camera = cv2.VideoCapture(0)



@app.route('/')
def home():
   return render_template('index.html')

imgb = np.zeros((480, 640, 3), dtype = np.uint8)


def gen():
    while True:
            success, image = camera.read()
            try:
                image=cv2.flip(image,1)
                imageBGR = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(imageBGR)
                image_height, image_width, c = image.shape
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:   
                        x_min,y_min,x_max,y_max=cal_box(hand_landmarks,image_shape=(image_height,image_width))
                        image= cv2.rectangle(image, (x_max+20,y_max+20), (x_min-20,y_min-20), (255, 0, 0), 2)
                        #print(x_max,y_max)
                        #img_=cv2.resize(data,(200,200))
                        mp_selfie_segmentation = mp.solutions.selfie_segmentation
                        segment = mp_selfie_segmentation.SelfieSegmentation(model_selection = 0)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = segment.process(image)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image_seg_mask = results.segmentation_mask
                        threshold = 0.5
                        binary_mask = image_seg_mask > threshold
                        mask3d = np.dstack((binary_mask, binary_mask, binary_mask))
                        replaced_img = np.where(mask3d, image, imgb)
                        #cv2.imwrite('temp.jpg',replaced_img[y_min-20:y_max+20,x_min-20:x_max+20])
                        prediction=predict(image[y_min-20:y_max+20,x_min-20:x_max+20])
                        image=cv2.putText(image,prediction,(x_max,y_max),fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
            except:
                  print("unable to read image")           

            ret, jpeg = cv2.imencode('.jpg', image)
            

            frame = jpeg.tobytes()
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def cal_box(handLadmark,image_shape):

    all_x, all_y = [], [] # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1])) # multiply x by image width
        all_y.append(int(handLadmark.landmark[hnd].y* image_shape[0])) # multiply y by image height

    return min(all_x), min(all_y), max(all_x), max(all_y) # return as (xmin, ymin, xmax, ymax)

def predict(image):
        letterpred = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']
        img_ = cv2.resize(image,(64, 64))  
        img_array = imag.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.
        
        prediction = model.predict(img_processed)
        
        index = np.argmax(prediction)
        print(index)
        return letterpred[index]
            
            
    

@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
   app.run()