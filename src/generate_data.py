import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from utils import *

def draw_count_frame(frame, cnt, total):
    text = "Frame: {}/{}".format(cnt, total)
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

def main():
    class_name = input("Input class name: ").replace(' ', '_')
    ls_landmark = []
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    for i in range(TIME_DELAY):
        print(i)
        time.sleep(1)

    while len(ls_landmark)<FRAMES:
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(1)==ord('q'):
                break
            
            # Convert to RGB and create pose estimation
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = mp_pose.Pose().process(rgb)

            # Draw and create data
            if pose_result.pose_landmarks:
                landmark = timestamp_landmark(pose_result)
                ls_landmark.append(landmark)
                frame = draw_landmark(frame, mp_draw, mp_pose, pose_landmarks=pose_result.pose_landmarks)

            # Draw frame count
            frame = draw_count_frame(len(ls_landmark), FRAMES, frame)

            # Show pose
            cv2.imshow('screen', frame)
    
    df = pd.DataFrame(ls_landmark)
    df.to_csv("../data/{}.csv".format(class_name), index=False)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()