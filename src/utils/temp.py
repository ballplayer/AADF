import os
from PIL import Image
import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity
import onnxruntime
import pandas as pd
import torch
from torch.nn import functional as F
from statistics import mean
import cv2


# 将像素值的每个预测类别分别编码为不同的颜色，然后将图像可视化
def decode_segmaps(image, label_colors, nc=7):
    # 函数将输出的2D图像，会将不同类编码为不同的颜色
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    rgb = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
    for i in range(nc):
        # for循环遍历所有的types
        r[image == i] = label_colors[i][0]
        g[image == i] = label_colors[i][1]
        b[image == i] = label_colors[i][2]
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype('uint8')


def raw_draw_save(raw_img_path, onnx_model_path, onnx_raw_draw_path):
    ImageNameList = []
    if not os.path.exists(onnx_raw_draw_path):
        os.makedirs(onnx_raw_draw_path)
    raw_img_list = sorted(os.listdir(raw_img_path))
    for i in range(len(raw_img_list)):
        raw_image = Image.open(raw_img_path + '/' + raw_img_list[i]).convert('RGB')
        raw_image = np.array(raw_image)  # (height, width, channels)，其中通道数为3（RGB三色）。
        img_1 = raw_image[::2, ::2, :]
        raw_image = img_1[::2, ::2, :]
        # 根据模型文件的输出信息得到置信度和分类
        pred_label, raw_prob, draw_image = seg_predict(onnx_model_path, raw_image)
        # 如果该图片中只识别到了背景，则不使用该图片
        if len(pred_label) == 1:
            for label in pred_label:
                if label == 0:
                    print("该图片中没有类别！")
                    continue
        ImageNameList.append(raw_img_list[i])
        length = len(pred_label)
        # 原始分割图存放位置，可修改
        cv2.imwrite(onnx_raw_draw_path + '/' + raw_img_list[i], draw_image[:, :, ::-1])
        # print("{}有{}个分类，分类结果为{}".format(raw_img_list[i], length, pred_label))
    return ImageNameList


def adv_draw_save(onnx_adv_draw_path, adv_path, onnx_model_path, ImageNameList, is_substitute=True):
    if not os.path.exists(onnx_adv_draw_path):
        os.makedirs(onnx_adv_draw_path)
    for i in range(len(ImageNameList)):
        adv_image = Image.open(adv_path + '/' + ImageNameList[i]).convert('RGB')
        new_image = np.array(adv_image)  # (height, width, channels)，其中通道数为3（RGB三色）。
        if is_substitute:
            img_1 = new_image[::2, ::2, :]
            new_image = img_1[::2, ::2, :]
        # 根据模型文件的输出信息得到置信度和分类
        pred_label, raw_prob, draw_image = seg_predict(onnx_model_path, new_image)
        length = len(pred_label)
        # 对抗样本分割图存放位置，可修改
        cv2.imwrite(onnx_adv_draw_path + '/' + ImageNameList[i], draw_image[:, :, ::-1])
        # print("{}有{}个分类，分类结果为{}".format(ImageNameList[i], length, pred_label))


def compute_seg_iou(seg1, seg2):
    classes1 = np.unique(seg1)  # 获取seg1中的类别
    classes2 = np.unique(seg2)  # 获取seg2中的类别
    # print(classes1, classes2)
    classes = np.union1d(classes1, classes2)  # 合并两个分割图中的类别
    classes = classes[classes != 0]  # 去掉背景标签0
    iou_per_class = []
    weights = []

    for class_id in classes:
        seg1_class = seg1 == class_id
        seg2_class = seg2 == class_id
        # print(seg1_class, seg2_class)
        intersection = np.logical_and(seg1_class, seg2_class)
        union = np.logical_or(seg1_class, seg2_class)

        intersection_count = np.sum(intersection)
        union_count = np.sum(union)

        if union_count == 0:
            iou = 0.0
        else:
            iou = intersection_count / union_count

        iou_per_class.append(iou)
        weights.append(np.sum(seg1_class))  # 使用seg1_class的像素数量作为权重

    weights = np.array(weights)
    weighted_iou = np.sum(np.array(iou_per_class) * weights) / np.sum(weights)

    return weighted_iou


# 计算扰动距离L2范数
def calculate_ssim(org_image, adv_image2):
    # 读取两幅图像
    img1 = img_as_float(org_image)
    img2 = img_as_float(adv_image2)

    # 计算 SSIM
    ssim = structural_similarity(img1, img2, channel_axis=2)
    ssim_normalized = (ssim + 1) / 2

    # print('Structural Similarity Index between the two images:', ssim, ssim_normalized)
    return ssim_normalized


def seg_output(model, image):
    c_image = image.copy()
    c_image = c_image.transpose((2, 0, 1))  # 通道数*高*宽
    c_image = c_image.astype(np.float32)
    c_image = c_image / 255.0
    # 增维
    image_t = c_image[np.newaxis, :]
    # 读取模型文件
    resnet_session = onnxruntime.InferenceSession(model)
    inputs = {resnet_session.get_inputs()[0].name: image_t}
    outs = resnet_session.run(None, inputs)[0]
    # 从输出信息中得到分类结果
    outputarg = np.argmax(outs, 1)[0]

    return outputarg


def seg_predict(model, image, label_colors=None):
    if label_colors is None:
        label_colors = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255]])
        # label_colors = np.array([[0, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255]])
    c_image = image.copy()
    c_image = c_image.transpose((2, 0, 1))
    c_image = c_image.astype(np.float32)
    # 对img_2中每个元素除以255，输入和输出维度均为（3，200，320）
    c_image = c_image / 255.0
    # 在axis为0的轴上扩充一个维度，输入维度为（3，200，320），输出维度为(1, 3, 200, 320)
    image_t = np.expand_dims(c_image, axis=0)

    # ===================使用onnx文件进行预测=======================
    resnet_session = onnxruntime.InferenceSession(model, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    inputs = {resnet_session.get_inputs()[0].name: image_t}
    outs = resnet_session.run(None, inputs)[0]

    out = torch.tensor(outs, dtype=torch.float64)
    # 将置信度映射到[0, 1]之间
    probs = F.softmax(out[0], dim=0).detach().numpy()
    #print("probs:", probs.shape)
    # 获取所有类别中的最大置信度
    probarg = probs.max(axis=0)
    # 获取每个像素的分类
    outputarg = np.argmax(outs, 1)[0]
    #print("outputarg:", outputarg.shape)
    # 类别通道转换成颜色通道，转换成一张rgb图像
    outputting = decode_segmaps(outputarg, label_colors)
    # 通过函数统计图片中总共有哪些分类
    pred = get_unique_numbers(outputarg)
    prob_mean = []
    # 获取每个分类的平均置信度
    for s in range(len(pred)):
        prob = []
        for i in range(outputarg.shape[0]):
            for j in range(outputarg.shape[1]):
                if pred[s] == outputarg[i][j]:
                    prob.append(probarg[i][j])
        prob_mean.append(mean(prob))

    return pred, prob_mean, outputting


def get_unique_numbers(array):
    unique_numbers = set()
    for row in array:
        for num in row:
            unique_numbers.add(num)
    return list(unique_numbers)


def evaluate(onnx_model_path, raw_images_path, analysis_path, adv_path, ImageNameList, is_substitute=True):
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    SSIM = []
    MIOU = []
    num_miou_lt_60 = 0
    Raw_Classes = []
    New_Classes = []
    ImageName = []
    num_png = len(ImageNameList)
    failure_num = 0

    for i in range(len(ImageNameList)):
        ImageName.append(ImageNameList[i])
        raw_image_path = raw_images_path + '/' + ImageNameList[i]
        adv_image_path = adv_path + '/' + ImageNameList[i]

        # 原始图片
        raw_image = Image.open(raw_image_path).convert('RGB')  # 改PIL
        raw_image = np.array(raw_image)  # (height, width, channels)，其中通道数为3（RGB三色）。
        img_1 = raw_image[::2, ::2, :]
        raw_image = img_1[::2, ::2, :]
        # 对抗样本
        adv_image = Image.open(adv_image_path).convert('RGB')
        adv_image = np.array(adv_image)  # (height, width, channels)，其中通道数为3（RGB三色）。
        if is_substitute:
            img_2 = adv_image[::2, ::2, :]
            adv_image = img_2[::2, ::2, :]

        pred_label, raw_prob, draw_image = seg_predict(onnx_model_path, raw_image)
        Raw_Classes.append(pred_label)
        new_pred, raw_prob, draw_image = seg_predict(onnx_model_path, adv_image)

        pre_output = seg_output(onnx_model_path, raw_image)
        pre_new_output = seg_output(onnx_model_path, adv_image)

        if pred_label == new_pred:
            failure_num += 1
            # print(ImageNameList[i], "攻击失败！")
            # SSIM.append(-1)
            # New_Classes.append(new_pred)
            # MIOU.append(-1)
            # continue

        iou = compute_seg_iou(pre_output, pre_new_output)
        # 如果iou为0，说明全是背景，不计入统计
        if 0 < iou < 0.6:
            num_miou_lt_60 += 1
        # print(ImageNameList[i], " mIou：", iou)
        MIOU.append(iou)
        New_Classes.append(new_pred)
        ssim = calculate_ssim(raw_image, adv_image)
        # print(ImageNameList[i], " ssim：", ssim)
        SSIM.append(ssim)

        analysis = [ImageName, SSIM, MIOU, Raw_Classes, New_Classes]
        # print('analysis', analysis)
        name = ['ImageName', 'SSIM', 'MIOU', 'RAW Classes', 'New Classes']
        df = pd.DataFrame(dict(zip(name, analysis)))
        # 数据导出为csv
        if is_substitute:
            df.to_csv(analysis_path + '/SegPGD_substitute.csv')
        else:
            df.to_csv(analysis_path + '/SegPGD_transform.csv')

    # print("class SUCCESS RATE:", (num_png - failure_num) / num_png)
    # print("mIou SUCCESS RATE:", num_miou_lt_60 / num_png)
    return (num_png - failure_num) / num_png,  num_miou_lt_60 / num_png
