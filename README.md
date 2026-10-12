# Real-Time Drowsiness Detector (CSC 126 Midterm Project)

## Project Overview
A computer vision application built with Python and OpenCV that monitors a driver's face and eyes via webcam, tracks eye-state behavior, and triggers an immediate warning alert if signs of prolonged eye closure are detected.

## OpenCV Techniques Used
1. **Image Operations / Analysis and Transformation:** Grayscale conversion, frame resizing, and rendering custom text/visual alert overlays.
2. **Haar Cascade Classifier:** Real-time detection of facial regions and eyes.
3. **Facial State Analysis:** Tracking eye visibility over consecutive frames to calculate closure duration.

## Setup Instructions
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
