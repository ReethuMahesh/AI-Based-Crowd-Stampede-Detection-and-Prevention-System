# AI-Based Crowd Stampede Detection and Prevention System

An AI-powered real-time crowd monitoring system designed to identify early signs of potentially dangerous crowd conditions from CCTV/video feeds and provide timely risk classification.

The system combines **YOLOv8-Pose for person detection and pose analysis, OpenCV-based optical flow for crowd movement analysis, 13 physical crowd features, and an SVM classifier** to classify crowd conditions into **SAFE, MEDIUM, DENSE, and HIGH** states.

### Key Features

* Real-time CCTV/video-based crowd analysis
* YOLOv8-Pose based person and fall detection
* Optical-flow-based movement and turbulence analysis
* 13 physical features for crowd-state assessment
* SVM-based crowd-state classification
* Safety guardrails for blind spots, close-up views, pavement/stadium conditions, and falls
* FastAPI and WebSocket-based real-time monitoring dashboard
* Experimental classification accuracy of **93.1%**

The project focuses on **early detection of abnormal crowd behavior and potential stampede conditions**, rather than relying only on crowd size or person counting.
