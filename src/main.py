import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from utils import *

from tensorflow.keras.models import load_model
import threading

def draw_label(frame, label):
    text = "Class: {}".format(label)
    pos = (10,30)
    scale = 1
    thickness = 2
    lineType = 2
    fontColor = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        text,
        pos,
        font,
        scale,
        fontColor,
        thickness,
        lineType
    )
    return frame

def detect(model, ls_landmark, classes):
    global label
    tensor = np.expand_dims(ls_landmark,axis=0)
    result = model.predict(tensor)
    label = classes[np.argmax(result[0])]
    print(np.round(np.array(result[0]),2))

label = ""

if __name__ == '__main__':
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    model = load_model('../models/model.h5')
    cap = cv2.VideoCapture(0)
    ls_landmark = []
    classes = load_classes()
    
    while True:
        ret, frame = cap.read()
        if (ret):
            if cv2.waitKey(1)==ord('q'):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = mp_pose.Pose().process(rgb)
            
            if pose_result.pose_landmarks:
                landmark = timestamp_landmark(pose_result)
                ls_landmark.append(landmark)
                frame = draw_landmark(frame, mp_draw, mp_pose, pose_result.pose_landmarks)
            
            if len(ls_landmark) == DATA_RANGE:
                t = threading.Thread(
                    target = detect,
                    args = (model, ls_landmark, classes)
                )
                t.start()
                ls_landmark = []
            frame = draw_label(frame, label)
            
            cv2.imshow('screen', frame)

cap.release()
cv2.destroyAllWindows()
    
    
    