import numpy as np
import torchvision.transforms as transforms
from PIL import Image

import os
import xml.etree.ElementTree as ET


FAIR1M = ['Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321', 'A220',
          'A330', 'A350', 'C919', 'ARJ21', 'other-airplane', 'Passenger Ship',
          'Motorboat', 'Fishing Boat', 'Tugboat', 'Engineering Ship',
          'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'other-ship',
          'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'Van', 'Trailer',
          'Tractor', 'Truck Tractor', 'Excavator', 'other-vehicle',
          'Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court',
          'Roundabout', 'Intersection', 'Bridge']

Sar_Ship_Dataset = ['ship']

FUSAR_Ship = ['BulkCarrier', 'CargoShip', 'ContainerShip', 'GeneralCargo', 'Dredger',
              'Fishing', 'LawEnforce', 'Other', 'Passenger', 'Reserved', 'Tanker',
              'Tanker-HazardB', 'Tug', 'Unspecified']

OpenSARShip = ['Cargo', 'Dredging', 'Fishing', 'Other Type', 'Tanker', 'Tug']

VOC2007 = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

VOC2012 = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush']

COCO_categories = [{'id': 1, 'name': 'person'},
 {'id': 2, 'name': 'bicycle'},
 {'id': 3, 'name': 'car'},
 {'id': 4, 'name': 'motorcycle'},
 {'id': 5, 'name': 'airplane'},
 {'id': 6, 'name': 'bus'},
 {'id': 7, 'name': 'train'},
 {'id': 8, 'name': 'truck'},
 {'id': 9, 'name': 'boat'},
 {'id': 10, 'name': 'traffic light'},
 {'id': 11, 'name': 'fire hydrant'},
 {'id': 13, 'name': 'stop sign'},
 {'id': 14, 'name': 'parking meter'},
 {'id': 15, 'name': 'bench'},
 {'id': 16, 'name': 'bird'},
 {'id': 17, 'name': 'cat'},
 {'id': 18, 'name': 'dog'},
 {'id': 19, 'name': 'horse'},
 {'id': 20, 'name': 'sheep'},
 {'id': 21, 'name': 'cow'},
 {'id': 22, 'name': 'elephant'},
 {'id': 23, 'name': 'bear'},
 {'id': 24, 'name': 'zebra'},
 {'id': 25, 'name': 'giraffe'},
 {'id': 27, 'name': 'backpack'},
 {'id': 28, 'name': 'umbrella'},
 {'id': 31, 'name': 'handbag'},
 {'id': 32, 'name': 'tie'},
 {'id': 33, 'name': 'suitcase'},
 {'id': 34, 'name': 'frisbee'},
 {'id': 35, 'name': 'skis'},
 {'id': 36, 'name': 'snowboard'},
 {'id': 37, 'name': 'sports ball'},
 {'id': 38, 'name': 'kite'},
 {'id': 39, 'name': 'baseball bat'},
 {'id': 40, 'name': 'baseball glove'},
 {'id': 41, 'name': 'skateboard'},
 {'id': 42, 'name': 'surfboard'},
 {'id': 43, 'name': 'tennis racket'},
 {'id': 44, 'name': 'bottle'},
 {'id': 46, 'name': 'wine glass'},
 {'id': 47, 'name': 'cup'},
 {'id': 48, 'name': 'fork'},
 {'id': 49, 'name': 'knife'},
 {'id': 50, 'name': 'spoon'},
 {'id': 51, 'name': 'bowl'},
 {'id': 52, 'name': 'banana'},
 {'id': 53, 'name': 'apple'},
 {'id': 54, 'name': 'sandwich'},
 {'id': 55, 'name': 'orange'},
 {'id': 56, 'name': 'broccoli'},
 {'id': 57, 'name': 'carrot'},
 {'id': 58, 'name': 'hot dog'},
 {'id': 59, 'name': 'pizza'},
 {'id': 60, 'name': 'donut'},
 {'id': 61, 'name': 'cake'},
 {'id': 62, 'name': 'chair'},
 {'id': 63, 'name': 'couch'},
 {'id': 64, 'name': 'potted plant'},
 {'id': 65, 'name': 'bed'},
 {'id': 67, 'name': 'dining table'},
 {'id': 70, 'name': 'toilet'},
 {'id': 72, 'name': 'tv'},
 {'id': 73, 'name': 'laptop'},
 {'id': 74, 'name': 'mouse'},
 {'id': 75, 'name': 'remote'},
 {'id': 76, 'name': 'keyboard'},
 {'id': 77, 'name': 'cell phone'},
 {'id': 78, 'name': 'microwave'},
 {'id': 79, 'name': 'oven'},
 {'id': 80, 'name': 'toaster'},
 {'id': 81, 'name': 'sink'},
 {'id': 82, 'name': 'refrigerator'},
 {'id': 84, 'name': 'book'},
 {'id': 85, 'name': 'clock'},
 {'id': 86, 'name': 'vase'},
 {'id': 87, 'name': 'scissors'},
 {'id': 88, 'name': 'teddy bear'},
 {'id': 89, 'name': 'hair drier'},
 {'id': 90, 'name': 'toothbrush'}]


def read_images(path, img_names):
    transform = transforms.Compose([
        transforms.Resize(640, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(640),
        transforms.ToTensor()
    ])

    imgs = []
    for img_name in img_names:
        img = Image.open(os.path.join(path, img_name))
        img = transform(img).numpy()
        imgs.append(img)
    imgs = np.array(imgs) * 255
    # imgs = np.array(imgs, dtype=np.float32)
    return imgs


def cls2index(cls_name, dataset_name, model_name):
    if model_name == 'FasterRCNN':
        if dataset_name == 'FAIR1M':
            return FAIR1M.index(cls_name) + 1
        elif dataset_name == 'Sar-Ship-Dataset':
            return Sar_Ship_Dataset.index(cls_name) + 1
        elif dataset_name == 'FUSAR-Ship':
            return FUSAR_Ship.index(cls_name) + 1
        elif dataset_name == 'OpenSARShip':
            return OpenSARShip.index(cls_name) + 1
        elif dataset_name == 'COCO':
            for i in COCO_categories:
                if i['name'] == cls_name:
                    return i['id']
    else:
        if dataset_name == 'FAIR1M':
            return FAIR1M.index(cls_name)
        elif dataset_name == 'Sar-Ship-Dataset':
            return Sar_Ship_Dataset.index(cls_name)
        elif dataset_name == 'FUSAR-Ship':
            return FUSAR_Ship.index(cls_name)
        elif dataset_name == 'OpenSARShip':
            return OpenSARShip.index(cls_name)
        elif dataset_name == 'COCO':
            return COCO.index(cls_name)


def read_labels(path, label_names, dataset_name, model_name):
    labels = []
    for label_name in label_names:
        tree = ET.parse(os.path.join(path, label_name))
        root = tree.getroot()
        label = []
        for obj in root.iter('object'):
            one_obj = []
            name = obj.find('name').text
            one_obj.append(cls2index(name, dataset_name, model_name))
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            one_obj.append(xmin)
            one_obj.append(ymin)
            one_obj.append(xmax)
            one_obj.append(ymax)
            label.append(one_obj)
        labels.append(label)
    return labels


def change_format(pred, dataset_name, model_name):
    labels, boxes, scores = pred[0], pred[1], pred[2]
    objs = []
    for i in range(len(labels)):
        obj = []
        obj.append(cls2index(labels[i], dataset_name, model_name))
        obj.append(scores[i])
        obj.append(boxes[i][0][0])
        obj.append(boxes[i][0][1])
        obj.append(boxes[i][1][0])
        obj.append(boxes[i][1][1])
        objs.append(obj)
    return objs


def extract_predictions(predictions_, conf_thresh, dataset_name, model_name):
    predictions_class = []
    if model_name == 'FasterRCNN':
        if dataset_name == "FAIR1M":
            predictions_class = [FAIR1M[i-1] for i in list(predictions_["labels"])]
        elif dataset_name == "Sar-Ship-Dataset":
            predictions_class = [Sar_Ship_Dataset[i-1] for i in list(predictions_["labels"])]
        elif dataset_name == "FUSAR-Ship":
            predictions_class = [FUSAR_Ship[i-1] for i in list(predictions_["labels"])]
        elif dataset_name == "OpenSARShip":
            predictions_class = [OpenSARShip[i-1] for i in list(predictions_["labels"])]
        elif dataset_name == "VOC2007":
            predictions_class = [VOC2007[i-1] for i in list(predictions_["labels"])]
        elif dataset_name == "COCO":
            for i in list(predictions_["labels"]):
                for j in COCO_categories:
                    if j['id'] == i:
                        predictions_class.append(j['name'])
    else:
        if dataset_name == "FAIR1M":
            predictions_class = [FAIR1M[i] for i in list(predictions_["labels"])]
        elif dataset_name == "Sar-Ship-Dataset":
            predictions_class = [Sar_Ship_Dataset[i] for i in list(predictions_["labels"])]
        elif dataset_name == "FUSAR-Ship":
            predictions_class = [FUSAR_Ship[i] for i in list(predictions_["labels"])]
        elif dataset_name == "OpenSARShip":
            predictions_class = [OpenSARShip[i] for i in list(predictions_["labels"])]
        elif dataset_name == "VOC2007":
            predictions_class = [VOC2007[i] for i in list(predictions_["labels"])]
        elif dataset_name == "COCO":
            predictions_class = [COCO[i] for i in list(predictions_["labels"])]

    if len(predictions_class) < 1:
        return [], [], []
    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    # print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = conf_thresh
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold]
    if len(predictions_t) > 0:
        predictions_t = predictions_t  # [-1] #indices where score over threshold
    else:
        # no predictions esxceeding threshold
        return [], [], []
    # predictions in score order
    predictions_boxes = [predictions_boxes[i] for i in predictions_t]
    predictions_class = [predictions_class[i] for i in predictions_t]
    predictions_scores = [predictions_score[i] for i in predictions_t]
    return predictions_class, predictions_boxes, predictions_scores



