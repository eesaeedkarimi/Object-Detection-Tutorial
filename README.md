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
2. [Datasets](#datasets)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Available Models](#available-models)
5. [Evaluation of the Results](#evaluation-of-the-results)
6. [Train the network and analyze errors](#train-the-network-and-analyze-errors)

## Problem definition
Object detection is the task of finding objects in an image. The type (class) and location of objects must also be determined. So the input is an image and the output is a list of objects each containing 6 numbers:  
```python
[Class, Location 1, Location 2, Location 3, Location 4, Score]
```

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

The result of an object detector is visualized in the following figure. **TODO**
 
## Datasets
Creating an object detection dataset is costly. Expert annotators have to draw the bounding boxes carefully around all 
of the objects that are defined in dataset classes in a way that the bounding box would be the smallest rectangle that 
includes all points of the object.  
Some of the most useful object detection datasets are as follows:  
### MS COCO
[Microsoft COCO](https://cocodataset.org/) is a large-scale object detection, segmentation, and captioning dataset. 
It is also a challenge that researchers evaluate their methods and the winners are rewarded based on their rank.  
Last version of the object detection dataset is released in 2017. The train, validation, and test sets, 
containing more than 200,000 images and 80 object classes, are available on the [download page of COCO](https://cocodataset.org/#download).

### Open Images
[Open Images](https://storage.googleapis.com/openimages/web/factsfigures.html) is the largest existing dataset of images 
annotated with labels of bounding boxes, segmentation masks, visual relationships, and localized narratives.
The last version of Open Images is V6 that is released in 2020. 
It contains a total of 16M bounding boxes for 600 object classes on 1.9M images and is available for free [download](https://storage.googleapis.com/openimages/web/download.html) 

### Pascal VOC
The [Pascal Visual Object Classes (VOC)](http://host.robots.ox.ac.uk/pascal/VOC/index.html) is a very popular dataset 
for image classification, object detection, and segmentation.
The last version of dataset is released in 2012. 
The train/val data has 11,530 images with 27,450 bounding box labels of 20 classes. It can be downloaded from the main 
website or it's [mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror).



### A survey on MS COCO
Every dataset has 3 main parts: Data, Labels and Evaluation scenarios. 
1. Data  
COCO is an image dataset so the Data part contains folders for [Train](http://images.cocodataset.org/zips/train2017.zip), 
[Validation](http://images.cocodataset.org/zips/val2017.zip), and [Test](http://images.cocodataset.org/zips/test2017.zip) images.  
COCO has an [Explore](https://cocodataset.org/#explore) 
page where image samples of each class can be seen with or without their labels.  
2. Labels
Bounding box labels of train and validation images are [available](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) as annotation files. 
Test images are used for challenge and do not have bounding box labels. 
However heir info is [available](http://images.cocodataset.org/annotations/image_info_test2017.zip).  
COCO labels are in JSON format. JSON is a file format that uses human-readable text to store data with 
attribute–value pairs and arrays.  
#### JSON files
Let's take a look at JSON format. An example of a JSON file is like:  

```json
{
    "attribute1": 1,
    "attribute2": [21, 22],
    "attribute3":
        [
            31,
            32,
            33
        ],
    "attribute4":
        [
            {
                "attribute4_1_1": 411,
                "attribute4_1_2": 412,
                "attribute4_1_3": 413
            },
            {
                "attribute4_2": 42
            }
        ]
}
```

Small size JSON files can be viewed by text editors like Notepad++. However they may not be as pretty formatted as above. 
For example the above JSON file can be like the following while it contains the same information:

```json
{"attribute1":1,"attribute2":[21,22],"attribute3":[31,32,33],"attribute4":[{"attribute4_1_1":411,"attribute4_1_2":412,"attribute4_1_3":413},{"attribute4_2":42}]}
```

Some plugins can also make pretty JSON files to show text editors. But it is not enough and some tools are needed to 
read, edit, and save JSON files in python. `json` library can perform these tasks.

As an example the following code uses `json` library to load the JSON file to a dictionary:
```python
import json
with open('./sample_files/sample_json.json', 'r') as f:
    data = json.load(f)
print(data['attribute1'])
print(data['attribute2'])
print(data['attribute3'])
print(data['attribute4'])
print(data['attribute4'][0]['attribute4_1_1'])
```
The result will be:
```
1
[21, 22]
[31, 32, 33]
[{'attribute4_1_1': 411, 'attribute4_1_2': 412, 'attribute4_1_3': 413}, {'attribute4_2': 42}]
411
```

Take a look at coco labels
train, val, test
load json of train file is too big can not be viewed by text editors.
main fields

field by field
number of images
number of annotations

categories

Evaluation codes (in next section)  

To download images: explore page of coco  
To download labels:  
read labels and explore  
json file, read json, what is in the file?
label format
number of images, what are labels? 

## Evaluation Metrics
mAP
Precision
Recall
IoU
Precision Recall curve
Mean over classes

## Available Models
yolov3

## Evaluation of the Results
main results
coco api
custom dataset
limit classes
limit classes and evaluate
we are going to fine tune and evaluate again

## Train the network and analyze errors
