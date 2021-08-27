import json

# ================================================================================ #
print('Exploring train annotation file')
with open('../COCO/annotations/instances_train2017.json', 'r') as f:
    train_annotations = json.load(f)

category_to_search = ['car', 'motorcycle', 'bicycle', 'bus', 'truck']
categories = train_annotations['categories']
category_ids = []
for category in categories:
    if category['name'] in category_to_search:
        category_ids.append(category['id'])

annotations = train_annotations['annotations']

any_sample_annotations = []
any_sample_image_ids = {}
any_sample_images = []
for annotation in annotations:
    category_id = annotation['category_id']
    if category_id in category_ids:
        any_sample_annotations.append(annotation)
        image_id = annotation['image_id']
        if image_id in any_sample_image_ids:
            any_sample_image_ids[image_id].add(category_id)
        else:
            any_sample_image_ids[image_id] = set([category_id])

all_sample_annotations = []
for id in any_sample_image_ids:

    if annotation['category_id'] in category_ids:
        any_sample_annotations.append(annotation)


print('End of Code')
