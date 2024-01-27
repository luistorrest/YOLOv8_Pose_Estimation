# -*- coding: utf-8 -*-
"""
This script uses YOLO (You Only Look Once) for pose detection and extracts keypoint information for yoga poses.
The keypoint information is then stored in a CSV file.

Make sure to install the required libraries using:
pip install ultralytics opencv-python matplotlib numpy pydantic

"""

from ultralytics import YOLO 
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
import os
import glob
import csv
import pandas as pd


# Define a class for keypoints
class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16
    
get_keypoint = GetKeypoint()


def extract_keypoint(keypoint):
    """
    Extract keypoint coordinates from the YOLO results.
    Args:
    - keypoint: List of keypoint coordinates.
    Returns:
    - List of keypoint coordinates.
    """

    # nose
    nose_x, nose_y = keypoint[get_keypoint.NOSE]
    # eye
    left_eye_x, left_eye_y = keypoint[get_keypoint.LEFT_EYE]
    right_eye_x, right_eye_y = keypoint[get_keypoint.RIGHT_EYE]
    # ear
    left_ear_x, left_ear_y = keypoint[get_keypoint.LEFT_EAR]
    right_ear_x, right_ear_y = keypoint[get_keypoint.RIGHT_EAR]
    # shoulder
    left_shoulder_x, left_shoulder_y = keypoint[get_keypoint.LEFT_SHOULDER]
    right_shoulder_x, right_shoulder_y = keypoint[get_keypoint.RIGHT_SHOULDER]
    # elbow
    left_elbow_x, left_elbow_y = keypoint[get_keypoint.LEFT_ELBOW]
    right_elbow_x, right_elbow_y = keypoint[get_keypoint.RIGHT_ELBOW]
    # wrist
    left_wrist_x, left_wrist_y = keypoint[get_keypoint.LEFT_WRIST]
    right_wrist_x, right_wrist_y = keypoint[get_keypoint.RIGHT_WRIST]
    # hip
    left_hip_x, left_hip_y = keypoint[get_keypoint.LEFT_HIP]
    right_hip_x, right_hip_y = keypoint[get_keypoint.RIGHT_HIP]
    # knee
    left_knee_x, left_knee_y = keypoint[get_keypoint.LEFT_KNEE]
    right_knee_x, right_knee_y = keypoint[get_keypoint.RIGHT_KNEE]
    # ankle
    left_ankle_x, left_ankle_y = keypoint[get_keypoint.LEFT_ANKLE]
    right_ankle_x, right_ankle_y = keypoint[get_keypoint.RIGHT_ANKLE]
    
    return [
        nose_x, nose_y,
        left_eye_x, left_eye_y,
        right_eye_x, right_eye_y,
        left_ear_x, left_ear_y,
        right_ear_x, right_ear_y,
        left_shoulder_x, left_shoulder_y,
        right_shoulder_x, right_shoulder_y,
        left_elbow_x, left_elbow_y,
        right_elbow_x, right_elbow_y,
        left_wrist_x, left_wrist_y,
        right_wrist_x, right_wrist_y,
        left_hip_x, left_hip_y,
        right_hip_x, right_hip_y,
        left_knee_x, left_knee_y,
        right_knee_x, right_knee_y,        
        left_ankle_x, left_ankle_y,
        right_ankle_x, right_ankle_y
    ]

# Initialize YOLO model
model = YOLO('yolov8n-pose.pt')

# Read an example image
img = cv2.imread('marcha.jpg')

# Define the dataset root directory
dataset_root = 'YogaPoses/Plank'

# Get the list of poses in the 'Plank' folder
pose_list = os.listdir(dataset_root)

# Create a list to store CSV data
dataset_csv = []

# Loop through each pose in the 'Plank' folder
for pose in pose_list:
    pose_path = os.path.join(dataset_root, pose)

    # Obtener la lista de imágenes en la pose actual
    #image_path_list = [os.path.join(pose_path, img) for img in os.listdir(pose_path) if img.endswith('.jpg')]


    # Get the name of the image
    image_name = os.path.basename(pose_path)

    # read image with OpenCV
    image = cv2.imread(pose_path)

    # Get the list of images in the current pose folder
    height, width = image.shape[:2]

    # Detect pose using YOLOv3-Pose
    results = model.predict(image, save=False)[0]
    results_keypoint = results[0].keypoints.xyn.cpu().numpy()

    for result_keypoint in results_keypoint:
        if len(result_keypoint) == 17:
            keypoint_list = extract_keypoint(result_keypoint)

            # Insert image name and pose label at positions 0 and 1
            keypoint_list.insert(0, image_name)
            keypoint_list.insert(1, pose)

             # Add keypoint list to the CSV dataset
            dataset_csv.append(keypoint_list)

## Write CSV
header = [
    'image_name',
    'label',
    # nose
    'nose_x',
    'nose_y',
    # left eye
    'left_eye_x',
    'left_eye_y',
    # right eye
    'right_eye_x',
    'right_eye_y',
    # left ear
    'left_ear_x',
    'left_ear_y',
    # right ear
    'right_ear_x',
    'right_ear_y',
    # left shoulder
    'left_shoulder_x',
    'left_shoulder_y',
    # right sholder
    'right_shoulder_x',
    'right_shoulder_y',
    # left elbow
    'left_elbow_x',
    'left_elbow_y',
    # rigth elbow
    'right_elbow_x',
    'right_elbow_y',
    # left wrist
    'left_wrist_x',
    'left_wrist_y',
    # right wrist
    'right_wrist_x',
    'right_wrist_y',
    # left hip
    'left_hip_x',
    'left_hip_y',
    # right hip
    'right_hip_x',
    'right_hip_y',
    # left knee
    'left_knee_x',
    'left_knee_y',
    # right knee
    'right_knee_x',
    'right_knee_y',
    # left ankle
    'left_ankle_x',
    'left_ankle_y',
    # right ankle
    'right_ankle_x',
    'right_ankle_y'
]

with open('yoga_pose_keypoint.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(dataset_csv)
    
# Read CSV using    

df = pd.read_csv('yoga_pose_keypoint.csv')
df = df.drop('image_name', axis=1)
df.head()


# results = model(source = img, conf=0.3)
# result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0]

# #xy = results[0].keypoints.xy

# # Se obtiene las coordenadas x,y del tensor a partir de la clase diseñada
#get_keypoint = GetKeypoint()
# nose_x, nose_y = result_keypoint[get_keypoint.NOSE]
# left_eye_x, left_eye_y = result_keypoint[get_keypoint.LEFT_EYE]
# left_ankle_x,  left_ankle_y = result_keypoint[get_keypoint.LEFT_ANKLE]

# key = extract_keypoint(result_keypoint)
