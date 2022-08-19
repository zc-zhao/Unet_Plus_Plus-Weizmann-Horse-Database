import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

#数据进行了归一化，将>0.5的为前景,<0.5的为背景
def get_miou(pre,mask):
    pre = torch.sigmoid(pre).cpu().data.numpy()
    mask = torch.sigmoid(mask).cpu().data.numpy()
    pre_ = pre > 0.5
    _pre = pre <= 0.5
    mask_ = mask > 0.5
    _mask = mask <= 0.5
    # 进行与或操作 获取交并集
    inter = (pre_ & mask_).sum()
    union = (pre_ | mask_).sum()
    _inter = (_pre & _mask).sum()
    _union = (_pre | _mask).sum()
    if union < 1e-5 or _union < 1e-5:
        return 0
    miou = (inter / union) * 0.5 + 0.5 * (_inter / _union)
    return miou

#利用腐蚀操作获取边界
def get_boundary(pic,ratio=0.02,sign=1):    #sign用于判断是预测图还是标签图
    if sign == 1:
        pic = torch.sigmoid(pic).data.cpu().numpy()
        pic[pic > 0.5] = 1
        pic[pic <= 0.5] = 0
        pic = pic.astype('uint8')
    elif sign == 0:
        pic = pic.cpu()
        pic = np.array(pic).astype('uint8')
        
    b, h, w = pic.shape
    new_pic = np.zeros([b, h + 2, w + 2])
    pic_erode = np.zeros([b, h, w])
    img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
    # 计算腐蚀参数dilation
    dilation = int(round(ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # 对一个batch中所有进行腐蚀操作
    for i in range(b):
        new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # 用0填充边框
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    for j in range(b):
        new_mask_erode = cv2.erode(new_pic[j], kernel, iterations=dilation)
        pic_erode[j] = new_mask_erode[1: h + 1, 1: w + 1]

    return pic - pic_erode

# 获取标签和预测的边界iou
def get_biou(mask, pre, dilation_ratio=0.02):
    pre_boundary = get_boundary(pre, dilation_ratio, sign=1)
    mask_boundary = get_boundary(mask, dilation_ratio, sign=0)
    B, H, W = pre_boundary.shape
    inter = 0
    union = 0
    # 计算交并比
    for k in range(B):
        inter += ((mask_boundary[k] * pre_boundary[k]) > 0).sum()
        union += ((mask_boundary[k] + pre_boundary[k]) > 0).sum()
    if union < 1:
        return 0
    boundary_iou = inter / union

    return boundary_iou
