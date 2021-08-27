import json

# ================================================================================ #
print('Exploring train annotation file')
with open('../COCO/annotations/instances_train2017.json', 'r') as f:
    train_annotations = json.load(f)

print('keys of train_annotations:')
print(train_annotations.keys())

print('========================================')
print('keys & values of Info:')
info = train_annotations['info']
for key in info.keys():
    print(f'{key}: {info[key]}')

print('========================================')
print('Licenses:')
licenses = train_annotations['licenses']
for license in licenses:
    print(license)

print('========================================')
print('Categories:')
categories = train_annotations['categories']
for category in categories:
    print(category)

print('========================================')
print('Images:')
images = train_annotations['images']
print(f'Number of train images: {len(images)}')

print('========================================')
print('Explore one image of train set:')
one_image = images[0]
for key in one_image.keys():
    print(f'{key}: {one_image[key]}')

print('========================================')
print('Annotations:')
annotations = train_annotations['annotations']
print(f'Number of train annotations: {len(annotations)}')

print('========================================')
print('Explore one annotation of train set:')
one_annotation = annotations[0]
for key in one_annotation.keys():
    print(f'{key}: {one_annotation[key]}')


# ================================================================================ #
print('========================================')
print('========================================')
print('Exploring validation annotation file')
with open('../COCO/annotations/instances_val2017.json', 'r') as f:
    val_annotations = json.load(f)

print('========================================')
print('Images:')
images = val_annotations['images']
print(f'Number of validation images: {len(images)}')

print('========================================')
print('Annotations:')
annotations = val_annotations['annotations']
print(f'Number of validation annotations: {len(annotations)}')

print('End of Code')
