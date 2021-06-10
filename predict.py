#!/usr/bin/env python
# encoding: utf-8
# @Time      :2021/6/6 14:53
# @Author    :Rakbow
# @File      :predict.py
import torch
import torchvision.transforms as transforms
from ssd.modeling.backbone.resnet_input_512 import ResNet
from PIL import Image


def predict_(img):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    # img =Image.open('E:\CLFAR-10+pyqt5\4.jpg')
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    model = ResNet()

    model_weight_pth = r'D:\Python\Python Project\SSD-Resnet-Attention-FeatureFusion\outputs\model_final.pth'
    model.load_state_dict(torch.load(model_weight_pth))

    model.eval()
    # classes = {'0': '飞机', '1': '汽车', '2': '鸟', '3': '猫', '4': '鹿', '5': '狗', '6': '青蛙', '7': '马', '8': '船', '9': '卡车'}
    classes = {'0': 'aeroplane', '1': 'bicycle', '2': 'bird', '3': 'boat',
               '4': 'bottle', '5': 'bus', '6': 'car', '7': 'cat', '8': 'chair',
               '9': 'cow', '10': 'diningtable', '11': 'dog', '12': 'horse',
               '13': 'motorbike', '14': 'person', '15': 'pottedplant',
               '16': 'sheep', '17': 'sofa', '18': 'train', '19': 'tvmonitor'}
    with torch.no_grad():
        output = torch.squeeze(model(img))
        print(output)
        predict = torch.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()

    return classes[str(predict_cla)], predict[predict_cla].item()
