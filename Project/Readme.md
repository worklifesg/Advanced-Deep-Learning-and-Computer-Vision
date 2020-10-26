## Project 

**Perform Facial Recognition with Deep Learning in Keras Using CNN**

![1](https://img.shields.io/badge/Platform-AWS%20SageMaker-orange) 
![2](https://img.shields.io/badge/Kernel-TensorFlow%202%20GPU%20Optimized-green) 
![3](https://img.shields.io/badge/Instance-4%20vCPU%20%2B%2016%20GiB%20%2B%201%20GPU-blue) 
![4](https://img.shields.io/badge/Dataset-ORL%20Faces-red)

### Description

**Problem Statement:** Facial recognition is a biometric alternative that measures unique characteristics of a human face. Applications available today include flight check in, tagging friends and family members in photos, and “tailored” advertising. You are a computer vision engineer who needs to develop a face recognition programme with deep convolutional neural networks.

**Objective:** Use a deep convolutional neural network to perform facial recognition using Keras.

**Dataset Details:** ORL face database composed of 400 images of size 112 x 92. There are 40 people, 10 images per person. The images were taken at different times, lighting and facial expressions. The faces are in an upright position in frontal view, with a slight left-right rotation.

### Details

* [Program Code]() + [Dataset]()

  * In the program, we analyzed ORL Faces Dataset where we already had train and test datasets. The image were regenarated using CNN model and accuracy is tested against test data.
  * The analysis for different activation functions is fisrt observed to find that 'leaky-relu' activation function is one of the activation functions that can be used for out final model.
  * The model training is done using x_train and y_train with validation data as x_valid and y_valid. owever for evaluating model, we use x_test and y_test which gives us loss ~0.2435 with an accuracy of 93.75%.
  
### Results (Figures)
