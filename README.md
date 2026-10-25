# AI-Driven Deepfake Detection — FaceForensics Benchmark Method

## Overview
This release contains the inference script and trained HOG + linear SVM model used for a FaceForensics benchmark prediction run.

## Method
1. Read images from the benchmark ZIP archive.
2. Convert each image to RGB and resize to 64 x 64 pixels.
3. Convert the image to grayscale.
4. Extract Histogram of Oriented Gradients (HOG) features.
5. Apply the trained classifier.
6. Map outputs to fake or real.
7. Write predictions to JSON.

The inference script does not train or tune the model.

## Files
- `predict_faceforensics_benchmark.py`: inference script
- `models/hog_linear_svm.joblib`: trained model
- `requirements.txt`: dependencies

## Installation
Use a compatible Python environment and run `pip install -r requirements.txt`.

## Input
The script expects `data/external/faceforensics_benchmark/faceforensics_benchmark_images.zip`. Obtain the dataset through the appropriate benchmark access process; it is not included here.

## Run
From the repository root, run `python predict_faceforensics_benchmark.py`.

Output: `reports/faceforensics_benchmark_predictions.json`.

## Prediction format
The output is a JSON object mapping image filenames to `fake` or `real`. Verify that filenames and labels match the current benchmark submission rules.

## Limitations
This documents the inference pipeline only. It does not claim benchmark acceptance or a particular performance score.
