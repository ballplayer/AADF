import os
import shutil
import time

import art.attacks.evasion as evasion_attack
import numpy as np
import torch
import torchvision
import yolov5
from PIL import Image
from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from pytorchyolo.models import load_model
from torchvision import transforms
from torchvision.utils import save_image as save_image_torch

from model.YOLO import YOLOv3, YOLOv5
from modeling.deeplab import DeepLab
from utils.RobotDataset import RobotDataset
from utils.Robot_utils import pgd
from utils.cls_idx2cls_name import idx2cls
from utils.create_fasterrcnn_model import create_model
from utils.object_detection import extract_predictions, read_images, change_format
from utils.save import save_adv, od_save_adv
from utils.temp import evaluate, adv_draw_save, raw_draw_save


def fb_attack_dataset(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device,
                      epsilons):
    file_content = open('record_content.txt', mode='a+')
    print(dataset_name, model_name, method_name)
    file_content.write(dataset_name + ' ' + model_name + ' ' + method_name + '\n')

    dir_dataset_name = os.path.join(save_path, dataset_name)
    dir_model_name = os.path.join(dir_dataset_name, model_name)
    dir_method_name = os.path.join(dir_model_name, method_name)
    # if distance_str is not None:
    #     dir_method_name = os.path.join(dir_method_name, 'distance={:}'.format(distance_str))
    #     print('distance={:}'.format(distance_str))
    #     file_content.write('distance={:}\n'.format(distance_str))
    for epsilon in epsilons:
        dir_epsilon = os.path.join(dir_method_name, 'epsilon={:.3f}'.format(epsilon))
        shutil.rmtree(dir_epsilon, ignore_errors=True)
        os.makedirs(dir_epsilon)

        predict_success_nums = 0
        attack_success_nums = 0
        attack_time = 0
        for batch_id, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            if method_name == 'DatasetAttack':
                if dataset_name == 'MSTAR':
                    target_image = Image.open('../data/MSTAR/test/SLICY/HB15214.JPG')
                elif dataset_name == 'ImageNet':
                    target_image = Image.open('../data/ImageNet/test/n01514859/ILSVRC2012_val_00011171.JPEG')
                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()
                                                ])
                target_image = transform(target_image).to(device)
                attack.feed(model=ae_model, inputs=torch.stack([target_image for _ in range(len(images))], dim=0))

            # 首先要判断是否预测正确，只有预测正确的样本才能对其进行攻击
            img_predict = torch.argmax(ae_model(images), dim=1)

            predict_true = (img_predict == labels)
            predict_success_nums += predict_true.sum().item()  # 预测正确的样本数
            start_time = time.time()
            adv, clipped, is_adv = attack(ae_model, images, labels, epsilons=epsilon)
            end_time = time.time()
            attack_time += (end_time - start_time)
            adv_predict = torch.argmax(ae_model(clipped), dim=1)

            # 攻击成功的样本应为：预测正确且is_adv为true的样本
            attack_success = predict_true & is_adv
            if attack_success.sum().item() > 0:
                print("labels:     {:}".format(idx2cls(dataset_name, labels)))
                file_content.write("labels:     {:}\n".format(idx2cls(dataset_name, labels)))
                print("img_predict:{:}".format(idx2cls(dataset_name, img_predict)))
                file_content.write("img_predict:{:}\n".format(idx2cls(dataset_name, img_predict)))
                print("adv_predict:{:}".format(idx2cls(dataset_name, adv_predict)))
                file_content.write("adv_predict:{:}\n".format(idx2cls(dataset_name, adv_predict)))

                images = images.cpu()
                save_adv(images, attack_success, clipped, dir_epsilon, attack_success_nums, file_content)
                attack_success_nums += attack_success.sum().item()

        print_result(epsilon, attack_success_nums, predict_success_nums, attack_time, len(data_loader), file_content)


def art_attack_dataset(model_name, data_loader, batch_size, dataset_name, save_path, method_name, ae_model, epsilons,
                       targeted, y_target, max_iter):
    global attack, adv, adv_predict_true, end_time, guide_label
    file_content = open('record_content.txt', mode='a+')
    print(dataset_name, model_name, method_name)
    file_content.write(dataset_name + ' ' + model_name + ' ' + method_name + '\n')

    dir_dataset_name = os.path.join(save_path, dataset_name)
    dir_model_name = os.path.join(dir_dataset_name, model_name)
    dir_method_name = os.path.join(dir_model_name, method_name)
    # if targeted is not None:
    #     dir_method_name = os.path.join(dir_method_name, 'targeted={:}'.format(targeted))
    #     print('targeted={:}'.format(targeted))
    #     file_content.write('targeted={:}\n'.format(targeted))
    for epsilon in epsilons:
        if epsilon > 0:
            dir_epsilon = os.path.join(dir_method_name, 'epsilon={:.3f}'.format(epsilon))
        else:
            epsilon = None
            dir_epsilon = dir_method_name

        shutil.rmtree(dir_epsilon, ignore_errors=True)
        os.makedirs(dir_epsilon)

        if method_name == 'AutoAttack':
            epsilon = (0.3 if epsilon is None else epsilon)
            attack = evasion_attack.AutoAttack(estimator=ae_model, batch_size=batch_size, eps=epsilon,
                                               targeted=targeted)
        elif method_name == 'AutoProjectedGradientDescent':
            epsilon = (0.3 if epsilon is None else epsilon)
            attack = evasion_attack.AutoProjectedGradientDescent(estimator=ae_model, batch_size=batch_size,
                                                                 eps=epsilon, targeted=targeted, verbose=False)
        elif method_name == 'AutoConjugateGradient':
            epsilon = (0.3 if epsilon is None else epsilon)
            attack = evasion_attack.AutoConjugateGradient(estimator=ae_model, batch_size=batch_size,
                                                          eps=epsilon, targeted=targeted, verbose=False)
        elif method_name == 'CarliniL0Method':
            attack = evasion_attack.CarliniL0Method(classifier=ae_model, batch_size=batch_size, targeted=targeted,
                                                    verbose=False)
        elif method_name == 'CarliniLInfMethod':
            attack = evasion_attack.CarliniLInfMethod(classifier=ae_model, batch_size=batch_size, targeted=targeted,
                                                      verbose=False)
        elif method_name == 'SaliencyMapMethod':
            attack = evasion_attack.SaliencyMapMethod(classifier=ae_model, batch_size=batch_size, verbose=False)
        elif method_name == 'GeoDA':
            max_iter = (4000 if max_iter is None else max_iter)
            attack = evasion_attack.GeoDA(estimator=ae_model, batch_size=batch_size, max_iter=max_iter, verbose=False)
        elif method_name == 'SquareAttack':
            epsilon = (0.3 if epsilon is None else epsilon)
            attack = evasion_attack.SquareAttack(estimator=ae_model, batch_size=batch_size, eps=epsilon, verbose=False)

        predict_success_nums = 0
        attack_success_nums = 0
        attack_time = 0
        for batch_id, (images, labels) in enumerate(data_loader):
            # 首先要判断是否预测正确，只有预测正确的样本才能对其进行攻击
            img_predict = torch.argmax(torch.from_numpy(ae_model.predict(images)), dim=1)
            predict_true = (img_predict == labels)
            predict_success_nums += predict_true.sum().item()
            start_time = time.time()
            if not targeted:
                adv = attack.generate(x=images.numpy())
                end_time = time.time()
                adv_predict = torch.argmax(torch.from_numpy(ae_model.predict(adv)), dim=1)

                adv_predict_true = (adv_predict != img_predict)
            else:
                adv = attack.generate(x=images.numpy(), y=y_target)
                end_time = time.time()
                adv_predict = torch.argmax(torch.from_numpy(ae_model.predict(adv)), dim=1)
                print("adv_predict:{:}".format(idx2cls(dataset_name, adv_predict)))
                file_content.write("adv_predict:{:}\n".format(idx2cls(dataset_name, adv_predict)))
                target = (torch.from_numpy(y_target.argmax(axis=1)) if len(y_target.shape) == 2 else y_target)

                # 以下方法的有目标是指：如果图片预测为y_target类，则进行攻击；否则不攻击
                if method_name in ['AutoAttack', 'GeoDA', 'SquareAttack']:
                    adv_predict_true = (labels == target) & (adv_predict != target)
                # 以下方法的有目标是指：攻击后预测为y_target类，且之前的图片不是y_target类
                else:
                    adv_predict_true = (labels != target) & (adv_predict == target)

            attack_time += (end_time - start_time)
            attack_success = predict_true & adv_predict_true

            if attack_success.sum().item() > 0:
                print("labels:     {:}".format(idx2cls(dataset_name, labels)))
                file_content.write("labels:     {:}\n".format(idx2cls(dataset_name, labels)))
                print("img_predict:{:}".format(idx2cls(dataset_name, img_predict)))
                file_content.write("img_predict:{:}\n".format(idx2cls(dataset_name, img_predict)))
                print("adv_predict:{:}".format(idx2cls(dataset_name, adv_predict)))
                file_content.write("adv_predict:{:}\n".format(idx2cls(dataset_name, adv_predict)))

                images = images.cpu()
                save_adv(images, attack_success, adv, dir_epsilon, attack_success_nums, file_content)
                if targeted and method_name in ['AutoAttack', 'GeoDA', 'SquareAttack']:
                    predict_success_nums -= (labels == target).sum().item()
                attack_success_nums += attack_success.sum().item()

        print_result(epsilon, attack_success_nums, predict_success_nums, attack_time, len(data_loader), file_content)


def new_white(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device, epsilons,
              ae_model2, attack2):
    file_content = open('record_content.txt', mode='a+')
    print(dataset_name, model_name, method_name)
    file_content.write(dataset_name + ' ' + model_name + ' ' + method_name + '\n')

    dir_dataset_name = os.path.join(save_path, dataset_name)
    dir_model_name = os.path.join(dir_dataset_name, model_name)
    dir_method_name = os.path.join(dir_model_name, method_name)

    for epsilon in epsilons:
        dir_epsilon = os.path.join(dir_method_name, 'epsilon={:.3f}'.format(epsilon))
        shutil.rmtree(dir_epsilon, ignore_errors=True)
        os.makedirs(dir_epsilon)

        predict_success_nums = 0
        attack_success_nums = 0
        attack_time = 0
        for batch_id, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 首先要判断是否预测正确，只有预测正确的样本才能对其进行攻击
            img_predict = torch.argmax(ae_model(images), dim=1)
            predict_true = (img_predict == labels)
            predict_success_nums += predict_true.sum().item()  # 预测正确的样本数
            start_time = time.time()
            adv, clipped, is_adv = attack(ae_model, images, labels, epsilons=epsilon)
            adv2, clipped2, is_adv2 = attack2(ae_model2, clipped, labels, epsilons=epsilon)
            end_time = time.time()
            attack_time += end_time - start_time

            adv_predict = torch.argmax(ae_model2(clipped2), dim=1)
            # 攻击成功的样本应为：预测正确且is_adv2为true的样本
            attack_success = predict_true & is_adv2

            if attack_success.sum().item() > 0:
                print("labels:     {:}".format(idx2cls(dataset_name, labels)))
                file_content.write("labels:     {:}\n".format(idx2cls(dataset_name, labels)))
                print("img_predict:{:}".format(idx2cls(dataset_name, img_predict)))
                file_content.write("img_predict:{:}\n".format(idx2cls(dataset_name, img_predict)))
                print("adv_predict:{:}".format(idx2cls(dataset_name, adv_predict)))
                file_content.write("adv_predict:{:}\n".format(idx2cls(dataset_name, adv_predict)))

                images = images.cpu()
                save_adv(images, attack_success, clipped2, dir_epsilon, attack_success_nums, file_content)
                attack_success_nums += attack_success.sum().item()

        print_result(epsilon, attack_success_nums, predict_success_nums, attack_time, len(data_loader), file_content)


def new_black(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device, batch_size,
              epsilons, ae_model2):
    file_content = open('record_content.txt', mode='a+')
    print(dataset_name, model_name, method_name)
    file_content.write(dataset_name + ' ' + model_name + ' ' + method_name + '\n')

    dir_dataset_name = os.path.join(save_path, dataset_name)
    dir_model_name = os.path.join(dir_dataset_name, model_name)
    dir_method_name = os.path.join(dir_model_name, method_name)

    for epsilon in epsilons:
        dir_epsilon = os.path.join(dir_method_name, 'epsilon={:.3f}'.format(epsilon))
        shutil.rmtree(dir_epsilon, ignore_errors=True)
        os.makedirs(dir_epsilon)

        predict_success_nums = 0
        attack_success_nums = 0
        attack_time = 0
        for batch_id, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 首先要判断是否预测正确，只有预测正确的样本才能对其进行攻击
            img_predict = torch.argmax(ae_model(images), dim=1)
            predict_true = (img_predict == labels)
            predict_success_nums += predict_true.sum().item()

            start_time1 = time.time()
            adv, clipped, is_adv = attack(ae_model, images, labels, epsilons=epsilon)
            end_time1 = time.time()
            max_iter = 4000
            if dataset_name in ['MSTAR', 'ImageNet']:
                max_iter = 500
            attack2 = evasion_attack.GeoDA(estimator=ae_model2, batch_size=batch_size, max_iter=max_iter, verbose=False)
            start_time2 = time.time()
            adv2 = attack2.generate(x=clipped.cpu().numpy())
            end_time2 = time.time()
            attack_time += (end_time1 - start_time1) + (end_time2 - start_time2)

            adv_predict = torch.argmax(torch.from_numpy(ae_model2.predict(adv2)), dim=1)
            img_predict = img_predict.cpu()
            adv_predict_true = (adv_predict != img_predict)
            predict_true = predict_true.cpu()
            attack_success = predict_true & adv_predict_true
            if attack_success.sum().item() > 0:
                print("labels:     {:}".format(idx2cls(dataset_name, labels)))
                file_content.write("labels:     {:}\n".format(idx2cls(dataset_name, labels)))
                print("img_predict:{:}".format(idx2cls(dataset_name, img_predict)))
                file_content.write("img_predict:{:}\n".format(idx2cls(dataset_name, img_predict)))
                print("adv_predict:{:}".format(idx2cls(dataset_name, adv_predict)))
                file_content.write("adv_predict:{:}\n".format(idx2cls(dataset_name, adv_predict)))

                save_adv(images, attack_success, adv2, dir_epsilon, attack_success_nums, file_content)
                attack_success_nums += attack_success.sum().item()

        print_result(epsilon, attack_success_nums, predict_success_nums, attack_time, len(data_loader), file_content)


def new_physical_world(attack, model_name, data_loader, dataset_name, save_path, method_name, ae_model, device,
                       batch_size, epsilons, ae_model2, shape):
    file_content = open('record_content.txt', mode='a+')
    print(dataset_name, model_name, method_name)
    file_content.write(dataset_name + ' ' + model_name + ' ' + method_name + '\n')
    dir_dataset_name = os.path.join(save_path, dataset_name)
    dir_model_name = os.path.join(dir_dataset_name, model_name)
    dir_method_name = os.path.join(dir_model_name, method_name)

    for epsilon in epsilons:
        dir_epsilon = os.path.join(dir_method_name, 'epsilon={:.3f}'.format(epsilon))
        shutil.rmtree(dir_epsilon, ignore_errors=True)
        os.makedirs(dir_epsilon)

        predict_success_nums = 0
        attack_success_nums = 0
        attack_time = 0
        for batch_id, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 首先要判断是否预测正确，只有预测正确的样本才能对其进行攻击
            img_predict = torch.argmax(ae_model(images), dim=1)
            predict_true = (img_predict == labels)
            predict_success_nums += predict_true.sum().item()

            start_time1 = time.time()
            adv, clipped, is_adv = attack(ae_model, images, labels, epsilons=epsilon)
            end_time1 = time.time()

            attack2 = evasion_attack.AdversarialPatch(classifier=ae_model2, batch_size=batch_size, patch_shape=shape,
                                                      targeted=False, verbose=False)
            start_time2 = time.time()
            # patch, patch_mask = attack2.generate(x=clipped.cpu().numpy())
            adv2 = attack2.apply_patch(clipped.cpu(), scale=0.3)
            end_time2 = time.time()
            attack_time += (end_time1 - start_time1) + (end_time2 - start_time2)

            adv_predict = torch.argmax(torch.from_numpy(ae_model2.predict(adv2)), dim=1)
            img_predict = img_predict.cpu()
            adv_predict_true = (adv_predict != img_predict)
            predict_true = predict_true.cpu()
            attack_success = predict_true & adv_predict_true
            if attack_success.sum().item() > 0:
                print("labels:     {:}".format(idx2cls(dataset_name, labels)))
                file_content.write("labels:     {:}\n".format(idx2cls(dataset_name, labels)))
                print("img_predict:{:}".format(idx2cls(dataset_name, img_predict)))
                file_content.write("img_predict:{:}\n".format(idx2cls(dataset_name, img_predict)))
                print("adv_predict:{:}".format(idx2cls(dataset_name, adv_predict)))
                file_content.write("adv_predict:{:}\n".format(idx2cls(dataset_name, adv_predict)))

                save_adv(images, attack_success, adv2, dir_epsilon, attack_success_nums, file_content)
                attack_success_nums += attack_success.sum().item()

        print_result(epsilon, attack_success_nums, predict_success_nums, attack_time, len(data_loader), file_content)


def print_result(epsilon, attack_success_nums, predict_success_nums, attack_time, length, file_content):
    if epsilon is None:
        print("epsilon:{:}  success rate:{:.3f}%  time(s)/batch:{:.3f}".
              format(epsilon, attack_success_nums / predict_success_nums * 100, attack_time / length))
        file_content.write("epsilon:{:}  success rate:{:.3f}%  time(s)/batch:{:.3f}\n".
                           format(epsilon, attack_success_nums / predict_success_nums * 100, attack_time / length))
    else:
        print("epsilon:{:.3f}  success rate:{:.3f}%  time(s)/batch:{:.3f}".
              format(epsilon, attack_success_nums / predict_success_nums * 100, attack_time / length))
        file_content.write("epsilon:{:.3f}  success rate:{:.3f}%  time(s)/batch:{:.3f}\n".
                           format(epsilon, attack_success_nums / predict_success_nums * 100, attack_time / length))


def od_attack(dataset_name, model_name, method_name, save_path, save_boxes_path):
    save_path = os.path.join(save_path, dataset_name, model_name, method_name)
    save_boxes_path = os.path.join(save_boxes_path, dataset_name, model_name, method_name)
    file_content = open('record_content.txt', mode='a+')

    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path)
    shutil.rmtree(save_boxes_path, ignore_errors=True)
    os.makedirs(save_boxes_path)

    input_shape = (3, 640, 640)
    clip_values = (0, 255)
    learning_rate = 1.99
    batch_size = 8
    max_iter = 20
    if method_name == 'ProjectedGradientDescent':
        max_iter = 10
    patch_shape = (3, 200, 200)
    conf_thresh = 0.5

    # 读取图片
    if dataset_name == 'COCO':
        imgs_list = os.listdir('../data/COCO/val50')
        imgs = read_images('../data/COCO/val50', imgs_list)
    elif dataset_name == 'VOC2007':
        imgs_list = os.listdir('../data/VOCdevkit/VOC2007/voc/JPEGImages50')
        imgs = read_images('../data/VOCdevkit/VOC2007/voc/JPEGImages50', imgs_list)
    elif dataset_name == 'FAIR1M':
        imgs_list = os.listdir('../data/FAIR1M/voc/JPEGImages50')
        imgs = read_images('../data/FAIR1M/voc/JPEGImages50', imgs_list)
    elif dataset_name == 'Sar-Ship-Dataset':
        imgs_list = os.listdir('../data/Sar-Ship-Dataset/ship_dataset_v0/voc/JPEGImages50')
        imgs = read_images('../data/Sar-Ship-Dataset/ship_dataset_v0/voc/JPEGImages50', imgs_list)
    elif dataset_name == 'FUSAR-Ship':
        imgs_list = os.listdir('../data/FUSAR-Ship/voc/JPEGImages50')
        imgs = read_images('../data/FUSAR-Ship/voc/JPEGImages50', imgs_list)
    elif dataset_name == 'OpenSARShip':
        imgs_list = os.listdir('../data/OpenSARShip/voc/JPEGImages50')
        imgs = read_images('../data/OpenSARShip/voc/JPEGImages50', imgs_list)
    else:
        print("No {:} dataset! Please select again!".format(dataset_name))
        exit(0)

    # 加载模型
    if model_name == 'YOLOv3':
        if dataset_name == 'FAIR1M':
            model = load_model(model_path="../model-cfg/FAIR1M-yolov3.cfg",
                               weights_path="../model-weights/FAIR1M-yolov3.pth")
        elif dataset_name == 'Sar-Ship-Dataset':
            model = load_model(model_path="../model-cfg/Sar-Ship-Dataset-yolov3.cfg",
                               weights_path="../model-weights/Sar-Ship-Dataset-yolov3.pth")
        elif dataset_name == 'FUSAR-Ship':
            model = load_model(model_path="../model-cfg/FUSAR-Ship-yolov3.cfg",
                               weights_path="../model-weights/FUSAR-Ship-yolov3.pth")
        elif dataset_name == 'OpenSARShip':
            model = load_model(model_path="../model-cfg/OpenSARShip-yolov3.cfg",
                               weights_path="../model-weights/OpenSARShip-yolov3.pth")
        elif dataset_name == 'VOC2007':
            model = load_model(model_path="../model-cfg/VOC2007-yolov3.cfg",
                               weights_path="../model-weights/VOC2007-yolov3.pt")
        elif dataset_name == 'COCO':
            model = load_model(model_path="../model-cfg/yolov3.cfg",
                               weights_path="../model-weights/yolov3.weights")

        model.to('cpu')
        model = YOLOv3(model)
        ae_model = PyTorchYolo(model=model,
                               device_type='cpu',
                               input_shape=input_shape,
                               clip_values=clip_values,
                               attack_losses=("loss_total", "loss_cls", "loss_box", "loss_obj")
                               )
    elif model_name == 'YOLOv5s':
        if dataset_name == 'FAIR1M':
            model = yolov5.load('../model-weights/FAIR1M-yolov5s.pt')
        elif dataset_name == 'Sar-Ship-Dataset':
            model = yolov5.load('../model-weights/Sar-Ship-Dataset-yolov5s.pt')
        elif dataset_name == 'FUSAR-Ship':

            model = yolov5.load('../model-weights/FUSAR-Ship-yolov5s.pt')
        elif dataset_name == 'OpenSARShip':
            model = yolov5.load('../model-weights/OpenSARShip-yolov5s.pt')
        elif dataset_name == 'VOC2007':
            model = yolov5.load('../model-weights/VOC2007-yolov5s.pt')
        elif dataset_name == 'COCO':
            model = yolov5.load('../model-weights/yolov5s.pt')

        model.to('cpu')
        model = YOLOv5(model)
        ae_model = PyTorchYolo(model=model,
                               device_type='cpu',
                               input_shape=input_shape,
                               clip_values=clip_values,
                               attack_losses=("loss_total", "loss_cls", "loss_box", "loss_obj")
                               )
    elif model_name == 'FasterRCNN':
        if dataset_name == 'FAIR1M':
            checkpoint = torch.load('../model-weights/FAIR1M-fasterrcnn.pth', map_location='cpu')
        elif dataset_name == 'Sar-Ship-Dataset':
            checkpoint = torch.load('../model-weights/Sar-Ship-Dataset-fasterrcnn.pth', map_location='cpu')
        elif dataset_name == 'FUSAR-Ship':
            checkpoint = torch.load('../model-weights/FUSAR-Ship-fasterrcnn.pth', map_location='cpu')
        elif dataset_name == 'OpenSARShip':
            checkpoint = torch.load('../model-weights/OpenSARShip-fasterrcnn.pth', map_location='cpu')
        elif dataset_name == 'VOC2007':
            checkpoint = torch.load('../model-weights/VOC2007-fasterrcnn.pth', map_location='cpu')
        elif dataset_name == 'COCO':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
            model.load_state_dict(torch.load('../model-weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'))
            # checkpoint = torch.load('../model-weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth', map_location='cpu')
        if dataset_name in ['FAIR1M', 'Sar-Ship-Dataset', 'FUSAR-Ship', 'OpenSARShip', 'VOC2007']:
            NUM_CLASSES = checkpoint['config']['NC']
            build_model = create_model[checkpoint['model_name']]
            model = build_model(num_classes=NUM_CLASSES, coco_model=False,
                                model_path='../model-weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to('cpu')
        model.eval()
        ae_model = PyTorchFasterRCNN(model=model,
                                     device_type='cpu',
                                     input_shape=input_shape,
                                     clip_values=clip_values,
                                     channels_first=True,
                                     # attack_losses=("loss_total", "loss_cls", "loss_box", "loss_obj")
                                     )
    else:
        print("No {:} model! Please select again!".format(model_name))
        exit(0)

    print(dataset_name, model_name, method_name)
    file_content.write(dataset_name + ' ' + model_name + ' ' + method_name + '\n')

    # img_predict
    x = imgs.copy()
    img_predict = ae_model.predict(x)  # dict_keys(['boxes', 'labels', 'scores'])

    attack_time = 0
    if method_name in ['RobustDPatch', 'DPatch', 'ProjectedGradientDescent']:
        if method_name == 'RobustDPatch':  # (YOLOv3 cpu batch_size=4)(YOLOv5s cpu batch_size=4)
            attack = evasion_attack.RobustDPatch(estimator=ae_model, patch_shape=patch_shape, patch_location=(50, 50),
                                                 learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
                                                 verbose=False)
            x = imgs.copy()
            start_time = time.time()
            adv_images = attack.apply_patch(x)
            end_time = time.time()
            attack_time += end_time - start_time
            x = adv_images.copy()
            adv_predicts = ae_model.predict(x)
        elif method_name == 'DPatch':  # (YOLOv3 cpu batch_size=4)(YOLOv5s cpu batch_size=4)
            attack = evasion_attack.DPatch(estimator=ae_model, batch_size=batch_size, patch_shape=patch_shape,
                                           learning_rate=50., max_iter=max_iter)
            start_time = time.time()
            x = imgs.copy()
            adv_images = attack.apply_patch(x)
            end_time = time.time()
            attack_time += end_time - start_time
            x = adv_images.copy()
            adv_predicts = ae_model.predict(x)
        elif method_name == 'ProjectedGradientDescent':  # (YOLOv3 cpu batch_size=4)(YOLOv5s cpu batch_size=4)
            attack = evasion_attack.ProjectedGradientDescent(estimator=ae_model, batch_size=batch_size,
                                                             max_iter=max_iter)
            x = imgs.copy()
            start_time = time.time()
            adv_images = attack.generate(x) * 255
            end_time = time.time()
            attack_time += end_time - start_time
            x = adv_images.copy()
            adv_predicts = ae_model.predict(x)
    else:
        print("No {:} method! Please select again!".format(method_name))
        exit(0)

    # 获取改变格式之前、之后的预测结果
    preds = []
    preds_format = []
    adv_preds = []
    adv_preds_format = []
    for i in range(len(adv_predicts)):
        pred = extract_predictions(img_predict[i], conf_thresh, dataset_name, model_name)
        preds.append(pred)
        preds_format.append(change_format(pred, dataset_name, model_name))
        adv_pred = extract_predictions(adv_predicts[i], conf_thresh, dataset_name, model_name)
        adv_preds.append(adv_pred)
        adv_preds_format.append(change_format(adv_pred, dataset_name, model_name))

    # 统计攻击成功个数，保存图片
    attack_success = 0
    for i in range(len(preds_format)):
        if len(preds_format) != len(adv_preds_format):
            attack_success += 1
            od_save_adv(imgs[i], preds[i], attack_success, adv_images[i], adv_preds[i], save_path, save_boxes_path)
        else:
            for j in range(len(preds_format[i])):
                if preds_format[i][j] not in adv_preds_format[i]:
                    attack_success += 1
                    od_save_adv(imgs[i], preds[i], attack_success, adv_images[i], adv_preds[i], save_path, save_boxes_path)
                    break

    print("adv-imgs saved in {:}\nadv-imgs with anchors saved in {:}\nsuccess rate:{:.3f}%   time(s)/img:{:.3f}".
          format(save_path, save_boxes_path, attack_success / len(preds) * 100, attack_time / len(imgs)))
    file_content.write("adv-imgs saved in in {:}\n".format(save_path))
    file_content.write("adv-imgs with anchors saved in {:}\n".format(save_boxes_path))
    file_content.write("success rate:{:.3f}%  time(s)/img:{:.3f}\n".
                       format(attack_success / len(preds) * 100, attack_time / len(imgs)))

    record_site_text = 'record_site.txt'
    file = open(record_site_text, mode='w')
    file.write(save_boxes_path)
    file.close()


def seg_attack(dataset_name, model_name, method_name, save_path, save_draw_path, epsilons, device, substitute):
    file_content = open('record_content.txt', mode='a+')
    img_size = (200, 320)
    if substitute:
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = RobotDataset(root_dir='../data/' + dataset_name, img_dir='img/train', label_dir='ann/train',
                               transform=trans)
    else:
        trans = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
        dataset = RobotDataset(root_dir='../data/' + dataset_name, img_dir='img/train', label_dir='ann/train',
                               transform=trans, target_transform=transforms.Resize(img_size))
    batch_size = len(dataset)
    X, y = [], []
    for i in range(batch_size):
        X.append(dataset[i][0])  # dataset[num][0]是一个tensor，shape为[3, 200, 320]
        y.append(np.asarray(dataset[i][1]))  # dataset[num][1]是一个PIL.Image.Image，shape为[200, 320]
    X, y = torch.stack(X), torch.tensor(y).unsqueeze(
        1)  # X:torch.Size([20, 3, 200, 320])   y:torch.Size([20, 1, 200, 320])
    X = X.to(device)
    y = y.to(device)
    if substitute:
        # 加载替代模型
        model = DeepLab(backbone='resnet', output_stride=16, num_classes=7, sync_bn=False, freeze_bn=False)
        checkpoint = torch.load('../model-weights/' + model_name + '.pth.tar', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.to(device)
    else:
        # 加载转换模型
        model = torch.load('../model-weights/' + model_name + '.pth')
        model.eval()
        model.to(device)

    # onnx模型路径
    onnx_model_path = '../model-weights/seg_model.onnx'

    # 用onnx模型对原始样本进行推理并上色
    raw_img_path = '../data/' + dataset_name + '/img/train'
    onnx_raw_draw_path = save_draw_path + dataset_name + '/' + model_name + '/' + method_name + '/onnx_raw_draw'
    print("colored imgs inferred by onnx saved in: ", onnx_raw_draw_path)
    file_content.write("onnx_raw_draw_path: {:}\n".format(onnx_raw_draw_path))
    ImageNameList = raw_draw_save(raw_img_path, onnx_model_path, onnx_raw_draw_path)

    alpha = 100
    num_iter = 50
    attack_time = 0
    # asrs_class = []
    # asrs_miou = []
    for epsilon in epsilons:
        adv = []
        for i in range(len(ImageNameList)):
            X_i = X[i].unsqueeze(0)  # X_i [1, 3, 800, 1280]
            y_i = y[i].unsqueeze(0)  # y_i [1, 1, 800, 1280]
            start_time = time.time()
            delta1 = pgd(model, X_i, y_i, epsilon=epsilon, alpha=alpha, num_iter=num_iter)  # [1, 3, 800, 1280]
            adv_i = X_i.float() + delta1.float()  # [1, 3, 800, 1280]
            end_time = time.time()
            attack_time += end_time - start_time
            adv.append(adv_i)
        adv1 = torch.cat(adv, dim=0)  # [20, 3, 800, 1280]
        # 保存生成的对抗样本
        c_path = dataset_name + '/' + model_name + '/' + method_name + '/epsilon=' + '{:.3f}'.format(epsilon)
        adv_path = save_path + c_path
        print('adv-imgs saved in: {:}', adv_path)
        file_content.write("adv-imgs saved in: {:}\n".format(adv_path))

        if not os.path.exists(adv_path):
            os.makedirs(adv_path)
        for i in range(len(ImageNameList)):
            save_image_torch(adv1[i], adv_path + '/' + ImageNameList[i])

        # 将替代模型生成的对抗样本输入到onnx模型中进行推理并上色
        onnx_adv_draw_path = save_draw_path + c_path
        print("colored adv-imgs inferred by onnx saved in: {:}", onnx_adv_draw_path)
        file_content.write("colored adv-imgs inferred by onnx saved in: {:}\n".format(onnx_adv_draw_path))
        adv_draw_save(onnx_adv_draw_path, adv_path, onnx_model_path, ImageNameList, is_substitute=substitute)

        # 评价指标
        analysis_path = save_draw_path + c_path + '/analysis'
        asr_class, asr_miou = evaluate(onnx_model_path, raw_img_path, analysis_path, adv_path, ImageNameList,
                                       is_substitute=substitute)
        # asrs_class.append(asr_class)
        # asrs_miou.append(asr_miou)

        print("epsilon:{:.3f}%   time(s)/img:{:.3f}\nasr based on class: {:}\nasr based on miou: {:}".
              format(epsilon, attack_time / len(ImageNameList), asr_class, asr_miou))
        file_content.write("epsilon:{:.3f}   time(s)/img:{:.3f}".format(epsilon, attack_time / len(ImageNameList)))
        file_content.write("asr based on class: {:}\n".format(asr_class))
        file_content.write("asr based on miou: {:}\n".format(asr_miou))

        record_site_text = 'record_site.txt'
        file = open(record_site_text, mode='w')
        file.write(onnx_raw_draw_path)
        file.write(onnx_adv_draw_path)
        file.close()
