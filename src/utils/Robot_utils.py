# coding=utf-8
import torch
import numpy as np
from PIL import Image
from torch import nn
import os


def pgd(model, X, y, epsilon, alpha, num_iter):  # Untargetted Attack 1
    delta = torch.zeros_like(X, requires_grad=True)  # 返回一个与给定输入张量形状和数据类型相同，但所有元素都被设置为零的新张量
    trg = y.squeeze(1)

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss(ignore_index=255)(model(X + delta), trg.long())
        loss.backward()
        # if (t + 1) % 10 == 0:
        #     print('Loss after iteration {}: {:.2f}'.format(t + 1, loss.item()))
        delta.data = (delta + X.shape[0] * alpha * delta.grad.data).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return delta.detach()


def pgd_steep(model, X, y, epsilon, alpha, num_iter):  # Untargetted Attack 2
    delta = torch.zeros_like(X, requires_grad=True)
    trg = y.squeeze(1)

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss(ignore_index=255)(model(X + delta)['out'], trg.long())
        loss.backward()
        if (t + 1) % 10 == 0:
            print('Loss after iteration {}: {:.2f}'.format(t + 1, loss.item()))
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return delta.detach()
