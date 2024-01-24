# -*- coding: utf-8 -*-

#Import necessary libraries
from ultralytics import YOLO 
import cv2

# Load YOLO model with pose estimation weights
model = YOLO('yolov8n-pose.pt')

# Open a video capture object for the default camera (index 0)
cap = cv2.VideoCapture(0)

# Main loop to capture video frames and perform YOLO object detection
while(cap.isOpened):
    # Read a frame from the video capture
    success, frame = cap.read()

    if success:
        # Perform object detection using YOLO and save the results
        results = model(frame, save=True)
        
        # Get the annotated frame with bounding boxes and labels
        annotated_frame = results[0].plot()
        
        # Display the annotated frame with YOLO predictions
        cv2.imshow('Yolo', annotated_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()