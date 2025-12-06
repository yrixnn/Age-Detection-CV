# Age-Detection-CV
# Objective:
To build an age detector that can approximately guess the gender and age of the person (face) in a picture or through a live feed or an uploaded image.

# Background:
The goal of this project is to develop a system that predicts the age range of individuals in images or real-time video streams using pre-trained deep learning models integrated with OpenCV.
Existing works on age detection utilize either handcrafted features or deep learning models. Deep learning approaches, especially convolutional neural networks (CNNs), are trained on large datasets like IMDB-WIKI by using the VGG-16 architecture and are pretrained on ImageNet for image classification. The research stated the key factors to approach the challenges was to implement deep learned models from large data, robust face alignment, and expected value formulation for age regression. By using standard benchmarks and achieving state-of-the-art, they were able to validate the methods on results for both real and apparent age estimation (Rothe, Timofte, Van Gool, 2016).


# Dataset:
The age prediction model was trained on the IMDB-WIKI dataset. This dataset contains over 500,000 labeled images of faces with corresponding ages, making it one of the largest publicly available datasets for age estimation.
Age Ranges: The model predicts eight age ranges:
- [0-2], [4-6], [8-12], [15-20], [25-32], [38-43], [48-53], [60+].
- Preprocessing: Images are resized to 227x227 pixels and normalized before being passed to the model.

# Required Libraries
pip install sklearn matplotlib

# Required Models:
- face_proto = "opencv_face_detector.pbtxt"
- face_model = "opencv_face_detector_uint8.pb"
- age_proto = "age_deploy.prototxt"
- age_model = "age_net.caffemodel"

# Required imports:
- import cv2
- import numpy as np
- from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
- import matplotlib.pyplot as plt
- import seaborn as sns
- import os

# Output:
![Output Zendaya Image](output_zendaya.png)

