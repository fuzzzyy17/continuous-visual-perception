# Timelog
* Continuous visual perception for object recognition
* Faizan Mohamed
* 2255205
* Gerado Aragon Camarasa

## Guidance


* This file contains the time log for your project. It will be submitted along with your final dissertation.
* **YOU MUST KEEP THIS UP TO DATE AND UNDER VERSION CONTROL.**
* This timelog should be filled out honestly, regularly (daily) and accurately. It is for *your* benefit.
* Follow the structure provided, grouping time by weeks.  Quantise time to the half hour.


## Week 1
The idea is to use deep learning (or traditional machine learning) techniques to aggregate knowledge through time to successfully identify and locate an object under while changing the vision sensor position. The later will run concurrently as Baxter attempts to grasp the detected object

### 23 Sep 2019

* *4 hrs*: Read the project guidance notes with lecture

### 24 Sep 2019

* *0.5 hrs*: Meeting with supervisor. 

### 27 Sep 2019

* *1 hrs*: Read research papers, basic requirement gathering
* *0.5 hrs*: Setup basic GitHub repo

### 1 Oct 2019

* *8 hrs*: Attempted to setup ROS on Ubuntu partition. Had to upgrade hard drive beforehand to make space for extended Ubuntu partition however copying over old hard drive to new failed. Reinstalled windows from scratch.

### 2 Oct 2019

* *8 hrs*: Attempted to setup ROS on windows, however kept on encountering many different errors. 

## 3 Oct 2019

* *0.5 hrs*: Meeting with supervisor. 

## 8 Oct 2019

* *1 hr*: Installed VirtualBox, Ubuntu, and ROS.

## 17th Oct 2019

* *0.5 hrs*: Meeting with supervisor

## 22nd Oct 2019

* *5 hrs*: Researching networks. Did basic research on how neural networks work, MLPS vs CNNs, etc. 

## 31st Oct 2019

* *0.5 hrs*: Met with advisor. Asked about differences between YOLO and MASK-RCNN (i.e. faster vs more widespread). Proposed new network called detectron 2.0

## November week 1

* *10 hrs*: Began learning mathematic fundamentals behind more specific features of perceptron models i.e. loss function, forward pass etc.  

## November week 2

* *15 hrs*: Started creating own basic perceptron models using MNIST dataset, however models often worked well for training data and not as well on test data.

## November week 3

* *10 hrs*: Researched into overfitting and underfitting, and applied measures to basic model to mitigate the effects of overfitting by reducing the training error. Perceptron works with ~96% on MNIST test data. 

## November week 4

* *5 hrs*: Spent time learning how to utilise GPU to increase speed at which models are trained at, however difference is minimal at the moment in time due to small datasets.

## December week 1

* *5 hrs*: Installed ROS successfully using VirtualBox, however runs slowly due to lack of RAM. 

## December week 2

* *15 hrs*: Decided to deepned understanding about softmax functions and why it is used instead of the sigmoid function for multiset data. Learned theory how to tune hyperparameters to make network learn better/faster, and reasoning behind it (e.g. adjusting learning rate, more convolutional layers, increased kernel size etc.)

## December week 3

* *8 hrs*: Researched how pooling works in context of CNNs, including feature maps, receptive fields, padding etc.

## December week 4

* *5 hrs*: Read into why ReLu is used, as well as understanding how/why the vanishing/exploding gradient problem occurs.

## Janurary week 1

* *15 hrs*: Began development of CNN to classify CIFAR10 dataset. Learnt about dropout function and when/how to use it for more accurate results. Results peaked around 60%.

## Janurary week 2

* *10 hrs*: Researched methods to make CNN more accurate. Learnt about data augmentation and how it can be used to 'increase' dataset size (by rotating, shearing etc) without actually increasing number of images.

## Janurary week 3

* *8 hrs*: Read into RNNs and how they differ from CNNs. Started research about how LTSMs work and how to incorporate one into my existing MNIST network.

## Janurary week 4

* *15 hrs*: Created an LTSM model for the MNIST dataset with 98% accuracy. Read theory behind LTSM and designed basic logic gate model for future use.

## February week 1

* *20 hrs*: started work in YOLO implementation in PyTorch. Began work on a darknet network file for object detection.

## February week 2

* *22 hrs*: continued work on YOLO darknet network. Researched into transformations required to ensure an efficient network - started work on tranformations file in order to transform output tensors into one large tensor.

## February week 3

* * *: Transformed feature maps so that they can now be concatenated together to form one tensor that can be operated on at once. Previously was not possible since cannot concat feature maps of differing spatial dims. Finished forward pass and detection layer. Beginning work on implementing pre-trained weights.