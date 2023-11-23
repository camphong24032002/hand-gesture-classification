import os
import pandas as pd

FRAMES = 100
TIME_DELAY = 5

DATA_RANGE = 10
EPOCHS = 32
BATCH_SIZE = 64

def load_data():
    data_dir = '../data'
    data_files = os.listdir(data_dir)
    
    data = {}
    for path in data_files:
        class_name = path.split('.')[0]
        data[class_name] = pd.read_csv(os.path.join(data_dir, path))
    return data

def load_classes():
    data_dir = '../data'
    data_files = os.listdir(data_dir)
    
    classes = []
    for path in data_files:
        class_name = path.split('.')[0]
        classes.append(class_name)
    return sorted(classes)

# Create dataset of landmarks and timestamp
def timestamp_landmark(pose):
    result = []
    for lm in pose.pose_landmarks.landmark:
        result.append(lm.x)
        result.append(lm.y)
        result.append(lm.z)
        result.append(lm.visibility)
    return result

# Draw landmarks on image
def draw_landmark(frame, mp_draw, mp_pose, pose_landmarks=None):
    mp_draw.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame