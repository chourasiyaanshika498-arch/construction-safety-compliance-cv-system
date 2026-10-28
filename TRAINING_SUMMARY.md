# Training Summary

## Project

Construction Safety Compliance Computer Vision System

## Model

- Model: YOLOv8n
- Task: PPE and construction safety object detection
- Image size: 512
- Epochs: 100
- Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Final weights: `runs/train/ppe_yolov8n_100e_reasonable/weights/best.pt`

## Dataset

- Train images: 30,765
- Validation images: 8,814
- Test images: 4,423
- Number of classes: 14

## Validation Results

- mAP50: 0.7664
- mAP50-95: 0.4949
- mAP75: 0.5273

## Test Results

- mAP50: 0.7763
- mAP50-95: 0.4993
- mAP75: 0.5262

## Strong Classes

The model performs well on:

- Fall-Detected
- Gloves
- Goggles
- Hardhat
- Ladder
- Person
- NO-Goggles
- NO-Gloves
- NO-Hardhat

## Weakest Class

The weakest class is:

- NO-Safety Vest

This class should be improved in future work using better class balancing, more examples, or targeted retraining.

## Notes

This model is suitable for a formal portfolio-level construction safety computer vision system. The system supports image analysis, video analysis, camera snapshots, live camera mode, safety rule checking, risk scoring, reporting, and sample prediction visualization.
