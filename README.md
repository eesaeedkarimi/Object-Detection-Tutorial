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

## Aproach
Our approach is to:
* Define the object detection problem.
* Choose one of the available models and evaluate it on the dataset.
* Improve the model.
* Evaluate it again and analyze errors.

## Table of Contents
1. [Object detection problem definition](https://github.com/eesaeedkarimi/Object-Detection-Tutorial#object-detection-problem-definition)
2. Object detection Datasets
3. Object detection Evaluation Metrics
4. Available models for object detection
5. Evaluation of Results

## Object detection problem definition
Object detection is the task of finding objects in an image. The type (class) and location of objects must also be determined. So the input is an image and the output is a list of objects each containing 6 numbers:  
\[Class, Location 1, Location 2, Location 3, Location 4, Score\]

### Class:
Each object detector can detect a limited types of objects. For example a hypothetical object detector that is trained for traffic control can detect car, motorcycle, bicycle, bus and truck. A number from 0 to 4 is assigned to each of these 5 classes and the object detector machine must determine the class of each object with this output parameter.  
Number and definition of these classes is determined by the dataset that used for training the model and adding a new class to a trained model is not a simple task.
### Location:
Object detector should find the Bounding Box of each object that is the smallest rectangle includes all points of the object. Bounding Box can be defined with one of these methods:
left, top, width, height \[x, y, w, h\]
or
left, top, right, bottom \[x1, y1, x2, y2\]
These above 4 numbers are Location1, Location2, Location3 and Loaction4.
