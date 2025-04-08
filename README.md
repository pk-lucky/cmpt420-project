# Camera-Based Object Detection and Tracking System

This project implements a camera-based object detection and tracking system using the nuScenes dataset.

## Prerequisites

### 1. nuScenes Mini Dataset

First, you need to download the nuScenes mini dataset:

1. Visit the [nuScenes website](https://www.nuscenes.org/download) and create an account
2. Download the "v1.0-mini" dataset (approximately 5GB)
3. Extract the dataset to the project directory
4. Ensure the dataset folder is named `v1.0-mini` in the project root

### 2. Python Environment

This project requires Python 3.8+ and several dependencies with specific version requirements.

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the object detection and tracking system:

```bash
python run_camera_tracking.py
```

This will:
1. Load the nuScenes mini dataset
2. Process the first scene in the dataset
3. Detect objects in each frame using a pre-trained Faster R-CNN model
4. Track objects across frames using a Kalman filter-based tracker
5. Generate visualization images for both detections and tracks
6. Create a video showing the tracking results

## Output

The system generates the following outputs in the `camera_tracking_results` directory:
- Individual frame images with detections (`detections_XXX.jpg`)
- Individual frame images with tracked objects (`tracks_XXX.jpg`)
- A video file (`tracking.mp4`) showing the tracking results