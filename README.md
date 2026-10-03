# CodeAlpha Object Detection and Tracking

## 📌 Project Overview

This project is developed as part of the CodeAlpha Artificial Intelligence Internship.

The project implements real-time object detection and tracking using a webcam. It uses a pre-trained YOLO model to detect objects in video frames and Deep SORT to track detected objects across consecutive frames.

## 🚀 Features

- Real-time webcam video input
- Object detection using YOLOv8
- Bounding boxes around detected objects
- Object class labels
- Unique tracking IDs
- Object tracking across video frames using Deep SORT
- Confidence-based detection filtering
- OpenCV-based video processing

## 🛠️ Technologies Used

- Python
- OpenCV
- YOLOv8
- Ultralytics
- Deep SORT
- deep-sort-realtime
- PyTorch

## 📂 Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
│
├── app.py
├── requirements.txt
├── .gitignore
├── yolov8n.pt
└── venv/
