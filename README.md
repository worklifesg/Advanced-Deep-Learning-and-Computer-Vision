## Advanced-Deep-Learning-and-Computer-Vision

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

![1](https://img.shields.io/badge/Python-3.6%2C%203.8.3-green) ![2](https://img.shields.io/badge/Tensorflow-2.3.0-orange)

![1](https://img.shields.io/badge/Platform-AWS%20SageMaker-orange) 
![2](https://img.shields.io/badge/Kernel-TensorFlow%202%20GPU%20Optimized-green) 
![3](https://img.shields.io/badge/Instance-4%20vCPU%20%2B%2016%20GiB%20%2B%201%20GPU-blue) 

This is a course from AI Engineer program from SimpliLearn. In this course, we will be able to classify different advanced models of machine learning:
```
Boltzman machines/RBM/DBNs
Various types of Auto-Encoders
Different types of GAN models
```
Also, we will practvie on certain applications such as:
```
NST - Neural Style Transfer
StyleGANs
Various type o fother GANs
YOLO/OpenCV based project
AutoEncoders/GANs for data generation
```
**Additional focus:** Deep Learning Model Deployment, Distributed computing using Tensorflow (V2) / Keras, Introduction to Reinforcement Learning

Fundamental Definitions/Terms:
  * **Advanced Deep Learning:** It is a field of study that deals with the recent advancements in deep learning.
  * **Need of Advanced Deep Learning:** To stay up-to-date with the recent advancements happening in deep learning, there should be a dedicated field of study.
  * **Computer Vision:** Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.
  * **Applications of Computer vision:**
    ```
    Traffic Monitoring, Interpretation of High Resolution Computer Images
    Optical Character Recognition, Target Recognition, 3D Shape Reconstruction
    Face Detection
    ```

**Practice Exercises:**
  * Image Pre-processing (Data Manipulation) - [Week Folder](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/tree/main/Week%201), [Notebook](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%201/1_ImagePreprocessingOperations_matplotlib.ipynb)
  * RBM/Autoencoders- [Week Folder](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/tree/main/Week%202), [Notebook - RBM](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%202/RBM_Practice_MNIST.ipynb), [Notebook - Autoencoders Part 1](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%202/AdvancedDL_OpenCV_10Oct2020.ipynb), [Notebook - Autoencoders Part 2](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%202/AdvancedDL_OpenCV_11Oct2020.ipynb), [Dataset](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%202/train%20(1).zip)
  * Generative Adversarial Networks - I (GANs) - [Week Folder](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/tree/main/Week%202), [Notebook - GAN Part 1](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%202/AdvancedDL_OpenCV_11Oct2020_Part2.ipynb)
  * Generative Adversarial Networks - II (GANs) and Neural transfer / Object Detection - [Week Folder](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/tree/main/Week%203), [[Notebook 1 DCGAN CIFAR10](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%203/dcgan_cifar10.ipynb)], [[Generated Image](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%203/generated_plot_e050.png)], [[Output .h5 File](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%203/generator_model_050.h5)], [[Notebook 2 CGAN FASHION MNIST](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%203/cgan_fashion_mnist.ipynb)], [[Output .h5 File](https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/Week%203/cgan_generator.h5)]

**Assisted Practice Projects** (To be completed)

  * Build an Movie Recommendation System Using RBM
  * Use Variational Autoencoder with Tensorflow to generater images using MNIST dataset
  * Use Variational Autoencoder with Keras to generate images using the MNIST dataset
  
  * Use Keras or TensorFlow to build a deep generative model that will translate drawings of shoes to designs.
  * Use YOLO v3 pretrained model for object detection

**Final Project**

*Perform Facial Recognition with Deep Learning in Keras Using CNN*

![1](https://img.shields.io/badge/Platform-AWS%20SageMaker-orange) 
![2](https://img.shields.io/badge/Kernel-TensorFlow%202%20GPU%20Optimized-green) 
![3](https://img.shields.io/badge/Instance-4%20vCPU%20%2B%2016%20GiB%20%2B%201%20GPU-blue) 
![4](https://img.shields.io/badge/Dataset-ORL%20Faces-red)

*Description*

*Problem Statement:* Facial recognition is a biometric alternative that measures unique characteristics of a human face. Applications available today include flight check in, tagging friends and family members in photos, and “tailored” advertising. You are a computer vision engineer who needs to develop a face recognition programme with deep convolutional neural networks.

*Objective:* Use a deep convolutional neural network to perform facial recognition using Keras.

*Dataset Details:* ORL face database composed of 400 images of size 112 x 92. There are 40 people, 10 images per person. The images were taken at different times, lighting and facial expressions. The faces are in an upright position in frontal view, with a slight left-right rotation.

*Details*

* [Program Code]() + [Dataset]()

  * In the program, we analyzed ORL Faces Dataset where we already had train and test datasets. The image were regenarated using CNN model and accuracy is tested against test data.
  * The analysis for different activation functions is fisrt observed to find that 'leaky-relu' activation function is one of the activation functions that can be used for out final model.
  * The model training is done using x_train and y_train with validation data as x_valid and y_valid. owever for evaluating model, we use x_test and y_test which gives us loss ~0.2435 with an accuracy of 93.75%.
  
*Results (Figures)*

<p align="center">
  <img width="250" alt="java 8 and prio java 8  array review example" img align="center" src ="https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/images/dataset%20images.png">
</p> 

<p align="center">
  <img width="350" alt="java 8 and prio java 8  array review example" img align="center" src ="https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/images/Activation_results1.png">
  <img width="350" alt="java 8 and prio java 8  array review example" img align="center" src ="https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/images/Activation_results2.png">
</p> 

<p align="center">
  <img width="350" alt="java 8 and prio java 8  array review example" img align="center" src ="https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/images/Final_results1.png">
  <img width="350" alt="java 8 and prio java 8  array review example" img align="center" src ="https://github.com/worklifesg/Advanced-Deep-Learning-and-Computer-Vision/blob/main/images/Final_results2.png">
</p> 


