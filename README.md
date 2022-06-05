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

The result of an object detector is visualized in the following figure.  
  
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg/1024px-Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg">](https://en.wikipedia.org/wiki/Object_detection#/media/File:Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg/)
 
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
Every dataset has 3 main parts: Data, Labels and Evaluation Scenarios. 
+ **Data**  
COCO is an image dataset so the Data part contains folders for [Train](http://images.cocodataset.org/zips/train2017.zip), 
[Validation](http://images.cocodataset.org/zips/val2017.zip), and [Test](http://images.cocodataset.org/zips/test2017.zip) images.  
COCO has an [Explore](https://cocodataset.org/#explore) 
page where image samples of each class can be seen with or without their labels.  
+ **Labels**  
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

Some plugins can also make pretty JSON files to show in text editors. But it is not enough and some tools are needed to 
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
>>> 1
>>> [21, 22]
>>> [31, 32, 33]
>>> [{'attribute4_1_1': 411, 'attribute4_1_2': 412, 'attribute4_1_3': 413}, {'attribute4_2': 42}]
>>> 411
```

#### MS COCO Labels  
Now let's take a look at COCO labels. As mentioned before, train and validation sets have bounding box labels.
After downloading and extracting `instances_train2017.json` and `instances_val2017.json` files into `COCO/annotations` 
directory, the following codes can be used to explore the annotations:  
```python
import json

print('Exploring train annotation file')
with open('./COCO/annotations/instances_train2017.json', 'r') as f:
    train_annotations = json.load(f)

print(train_annotations.keys())
```
```
>>> dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
```
keys & values of Info:  
```python
info = train_annotations['info']
for key in info.keys():
    print(f'{key}: {info[key]}')
```
```
>>> description: COCO 2017 Dataset
>>> url: http://cocodataset.org
>>> version: 1.0
>>> year: 2017
>>> contributor: COCO Consortium
>>> date_created: 2017/09/01
```
Licenses:
```python
licenses = train_annotations['licenses']
for license in licenses:
    print(license)
```
```
>>> Licenses:
>>> {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}
>>> {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}
>>> {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}
>>> {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}
>>> {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}
>>> {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}
>>> {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}
>>> {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}
```
Categories:  
```python
categories = train_annotations['categories']
for category in categories:
    print(category)
```
```
>>> {'supercategory': 'person', 'id': 1, 'name': 'person'}
>>> {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
>>> {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
>>> {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}
>>> {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}
>>> {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}
>>> {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}
>>> {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}
>>> {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}
>>> {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}
>>> {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}
>>> {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}
>>> {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}
>>> {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}
>>> {'supercategory': 'animal', 'id': 16, 'name': 'bird'}
>>> {'supercategory': 'animal', 'id': 17, 'name': 'cat'}
>>> {'supercategory': 'animal', 'id': 18, 'name': 'dog'}
>>> {'supercategory': 'animal', 'id': 19, 'name': 'horse'}
>>> {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}
>>> {'supercategory': 'animal', 'id': 21, 'name': 'cow'}
>>> {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}
>>> {'supercategory': 'animal', 'id': 23, 'name': 'bear'}
>>> {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}
>>> {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}
>>> {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}
>>> {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}
>>> {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}
>>> {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}
>>> {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}
>>> {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}
>>> {'supercategory': 'sports', 'id': 35, 'name': 'skis'}
>>> {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}
>>> {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}
>>> {'supercategory': 'sports', 'id': 38, 'name': 'kite'}
>>> {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}
>>> {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}
>>> {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}
>>> {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}
>>> {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}
>>> {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}
>>> {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}
>>> {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}
>>> {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}
>>> {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}
>>> {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}
>>> {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}
>>> {'supercategory': 'food', 'id': 52, 'name': 'banana'}
>>> {'supercategory': 'food', 'id': 53, 'name': 'apple'}
>>> {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}
>>> {'supercategory': 'food', 'id': 55, 'name': 'orange'}
>>> {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}
>>> {'supercategory': 'food', 'id': 57, 'name': 'carrot'}
>>> {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}
>>> {'supercategory': 'food', 'id': 59, 'name': 'pizza'}
>>> {'supercategory': 'food', 'id': 60, 'name': 'donut'}
>>> {'supercategory': 'food', 'id': 61, 'name': 'cake'}
>>> {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}
>>> {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}
>>> {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}
>>> {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}
>>> {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}
>>> {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}
>>> {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}
>>> {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}
>>> {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}
>>> {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}
>>> {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}
>>> {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}
>>> {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}
>>> {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}
>>> {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}
>>> {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}
>>> {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}
>>> {'supercategory': 'indoor', 'id': 84, 'name': 'book'}
>>> {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}
>>> {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}
>>> {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}
>>> {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}
>>> {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}
>>> {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}
```
Images:  
```python
images = train_annotations['images']
print(f'Number of train images: {len(images)}')
```
```
>>> Number of train images: 118287
```
Explore one image of train set:
```python
one_image = images[0]
for key in one_image.keys():
    print(f'{key}: {one_image[key]}')
```
```
>>> license: 3
>>> file_name: 000000391895.jpg
>>> coco_url: http://images.cocodataset.org/train2017/000000391895.jpg
>>> height: 360
>>> width: 640
>>> date_captured: 2013-11-14 11:18:45
>>> flickr_url: http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg
>>> id: 391895
```
Annotations:  
```python
annotations = train_annotations['annotations']
print(f'Number of train annotations: {len(annotations)}')
```
```
>>> Number of train annotations: 860001
```
Explore one annotation of train set
```python
one_annotation = annotations[0]
for key in one_annotation.keys():
    print(f'{key}: {one_annotation[key]}')
```
```
>>> segmentation: [[239.97, 260.24, 222.04, 270.49, 199.84, 253.41, 213.5, 227.79, 259.62, 200.46, 274.13, 202.17, 277.55, 210.71, 249.37, 253.41, 237.41, 264.51, 242.54, 261.95, 228.87, 271.34]]
>>> area: 2765.1486500000005
>>> iscrowd: 0
>>> image_id: 558840
>>> bbox: [199.84, 200.46, 77.71, 70.88]
>>> category_id: 58
>>> id: 156
```

Each image may have more than one annotation. Every annotation has an `image_id` field that shows which image does this 
annotation belong to? Can you write a code to find all annotations of the image with `image_id=391895`?  
  
**Exploring validation annotation file:**
```python
with open('../COCO/annotations/instances_val2017.json', 'r') as f:
    val_annotations = json.load(f)
```
Images:
```python
images = val_annotations['images']
print(f'Number of validation images: {len(images)}')
```
```
>>> Number of validation images: 5000
```
Annotations:
```python
annotations = val_annotations['annotations']
print(f'Number of validation annotations: {len(annotations)}')
```
```
>>> Number of validation annotations: 36781
```

+ **Evaluation Scenarios**    
Every dataset should has a well defined evaluation scenario so that a researcher can compare the performance of 
different models or a model before and after fine tuning. In the next sections evaluation metrics and available tools for
 evaluations will be explored.  

## Evaluation Metrics
In this section 
mAP,
Precision,
Recall,
IoU,
Precision Recall curve, and
Mean over classes
will be described.  
## Available Models
In this section 
some available methods like 
yolov3 and efficientNet 
will be described.  
## Evaluation of the Results
In this section 
main results of a model, 
coco api, 
custom dataset, 
how to limit classes, 
evaluation of limited classes, 
will be described.  
## Train the network and analyze errors
In this section, we are going to fine tune and evaluate the model again.  
