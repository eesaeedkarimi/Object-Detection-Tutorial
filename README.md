# Object Detection Tutorial
In this tutorial we are going to see how to train and evaluate an object detection model on a custom or standard dataset.

Various trained models on standard datasets are available. But suppose you want to:  
* train your own model on a standard dataset and achieve better results  
or  
* train or fine tune one of the available models on a custom dataset.

In both cases, this tutorial helps you do this with a scientific approach.

What you need to know in advance or study from other sources:
* Python programming
* Working with deep learning frameworks

What we will explore deeply:
* Available models for object detection
* Standard datasets of object detection
* Evaluation metrics

## Approach
Our approach is to:
* Define the object detection problem.
* Choose one of the available models and evaluate it on the dataset.
* Improve the model.
* Evaluate it again and analyze errors.

## Table of Contents
1. [Problem definition](#problem-definition)
2. [Datasets](https://github.com/eesaeedkarimi/Object-Detection-Tutorial#datasets)
3. [Evaluation Metrics](https://github.com/eesaeedkarimi/Object-Detection-Tutorial#evaluation-metrics)
4. [Available Models](https://github.com/eesaeedkarimi/Object-Detection-Tutorial#available-models)
5. [Evaluation of the Results](https://github.com/eesaeedkarimi/Object-Detection-Tutorial#evaluation-of-the-results)
6. [Train the network and analyze errors](#train-the-network-and-analyze-errors)

## Problem definition
Object detection is the task of finding objects in an image. The type (class) and location of objects must also be determined. So the input is an image and the output is a list of objects each containing 6 numbers:  
\[Class, Location 1, Location 2, Location 3, Location 4, Score\]

### Class:
Each object detector can detect a limited types of objects. For example a hypothetical object detector that is trained 
for traffic control can detect car, motorcycle, bicycle, bus and truck. A number from 0 to 4 is assigned to each of 
these 5 classes and the object detector machine must determine the class of each object with this output parameter.  
Number and definition of these classes are determined by the dataset that is used for training the model and adding a 
new class to a trained model is not a simple task.
### Location:
Object detector should find the Bounding Box of each object that is the smallest rectangle includes all points of the 
object. Bounding Box can be defined with one of these methods:  
* left, top, width, height \[x, y, w, h\]  
or  
* left, top, right, bottom \[x1, y1, x2, y2\]

These above 4 numbers are Location1, Location2, Location3 and Loaction4.
### Score:
Score value which is a number between 0 and 1, defines how much the trained machine is sure that an object with this class exists in this location of image.

The result of an object detector is visualized in the following figure.
 
## Datasets
Creating an object detection dataset is costly. Expert annotators have to draw the bounding boxes carefully around all 
of the objects that are defined in dataset classes in a way that the bounding box would be the smallest rectangle that 
includes all points of the object.  
Some of the most useful object detection datasets are as follows:  
### MS COCO

### Open Images

### Pascal VOC

## Evaluation Metrics

## Available Models

## Evaluation of the Results

## Train the network and analyze errors