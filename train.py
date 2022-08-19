import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch import optim
import argparse
import time

from dataset import dataloder
from index import get_biou,get_miou
from Net import UnetPlusPlus
from draw import draw

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--imagef',default='./archive/weizmann_horse_db',help='folder to data images and masks') #数据集保存路径
parser.add_argument('--modelf', default='./model/', help='folder to model checkpoints')  # 模型参数保存路径
parser.add_argument('--outf', default='./output/', help='folder to output images and out file')  # 输出文件保存路径
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#BCE_DICE损失
def bce_dice_loss(pre_batch, mask_batch):
    smooth = 1e-5
    # BCE损失
    pre_batch = pre_batch.squeeze(dim=1)
    bce = F.binary_cross_entropy_with_logits(pre_batch, mask_batch)
    # DICE损失
    predict = torch.sigmoid(pre_batch)
    num = mask_batch.size(0)
    predict = predict.view(num, -1)  # torch展平
    mask_batch = mask_batch.view(num, -1)  # torch展平
    inter = (predict * mask_batch)
    dice = (2. * inter.sum(1) + smooth) / (predict.sum(1) + mask_batch.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return 0.5 * bce + dice

def train(model,Train_data,Test_data,epoches,lr,deep_supervision=False):
    #优化器用adam
    Optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    #余弦退火更改学习率
    scheduler = lr_scheduler.CosineAnnealingLR(Optimizer, T_max=epoches, eta_min=1e-5)
    
    TRAIN_MIOU=[]
    TRAIN_BIOU=[]
    TRAIN_LOSS=[]
    TEST_MIOU=[]
    TEST_BIOU=[]
    TEST_LOSS=[]
    
    for epoch in range(1,epoches+1):
        model.train()
        train_MIOU=[]
        train_BIOU=[]
        train_LOSS=[]
        print('\nepoch',epoch,':')
        print('\ntraining.......')
    
        for img_batch,mask_batch in Train_data:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            loss = 0
        # 判断是否采用深监督  批次输入and输出
            if deep_supervision:
                outputs = model(img_batch)
                for output in outputs:
                    loss += bce_dice_loss(output, mask_batch).cuda()
                loss /= len(outputs)
                miou = get_miou(outputs[-1].squeeze(dim=1), mask_batch)
                biou = get_biou(mask_batch, outputs[-1].squeeze(dim=1))
            else:
                output = model(img_batch)
                loss = bce_dice_loss(output, mask_batch)
                miou = get_miou(output.squeeze(dim=1), mask_batch)
                biou = get_biou(mask_batch, output.squeeze(dim=1))
        
            train_MIOU.append(miou)
            train_BIOU.append(biou)
            train_LOSS.append(loss)
            
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            
        print('train_miou',sum(train_MIOU) / len(train_MIOU))
        print('train_biou',sum(train_BIOU) / len(train_BIOU))
        print('train_loss',sum(train_LOSS) / len(train_LOSS))
        
        TRAIN_MIOU.append(sum(train_MIOU) / len(train_MIOU))
        TRAIN_BIOU.append(sum(train_BIOU) / len(train_BIOU))
        TRAIN_LOSS.append(sum(train_LOSS) / len(train_LOSS))
        
        
        model.eval()
        test_MIOU=[]
        test_BIOU=[]
        test_LOSS=[]
        
        print('\ntesting.......')
        with torch.no_grad():
            for test_img_batch, test_mask_batch in Test_data:
                test_img_batch = test_img_batch.to(device)
                test_mask_batch = test_mask_batch.to(device)
                test_loss = 0
                if deep_supervision:
                    test_outputs = model(test_img_batch)
                    for test_output in test_outputs:
                        test_loss += bce_dice_loss(test_output, test_mask_batch).cuda()
                    test_loss /= len(test_outputs)
                    test_miou = get_miou(test_outputs[-1].squeeze(dim=1), test_mask_batch)
                    test_biou = get_biou(test_mask_batch, test_outputs[-1].squeeze(dim=1))

                else:
                    test_output = model(test_img_batch)
                    test_loss += bce_dice_loss(test_output, test_mask_batch)
                    test_miou = get_miou(test_output.squeeze(dim=1), test_mask_batch)
                    test_biou = get_biou(test_mask_batch, test_output.squeeze(dim=1))

                test_MIOU.append(test_miou)
                test_BIOU.append(test_biou)
                test_LOSS.append(test_loss)
        
        scheduler.step()
        
        print('test_miou',sum(test_MIOU) / len(test_MIOU))
        print('test_biou',sum(test_BIOU) / len(test_BIOU))
        print('test_loss',sum(test_LOSS) / len(test_LOSS))
        
        TEST_MIOU.append(sum(test_MIOU) / len(test_MIOU))
        TEST_BIOU.append(sum(test_BIOU) / len(test_BIOU))
        TEST_LOSS.append(sum(test_LOSS) / len(test_LOSS))
        
        
    print('Saving model......')
    torch.save(model.state_dict(), '%s/%s.pth' % (args.modelf, 'Unet_Plus_Plus_model'))
    
    return TRAIN_MIOU,TRAIN_BIOU,TRAIN_LOSS,TEST_MIOU,TEST_BIOU,TEST_LOSS


if __name__ == '__main__':
    print('-'*33)
    deep_supervision=True       #是否采用深监督
    ratio=0.15                  #测试集占比
    root=args.imagef            #数据集存放路径
    png_root=args.outf          #数据结果存放路径
    epoches=50
    lr=1e-3
    batches=8
    imagesize=80                #照片大小
    
    Train_data,Test_data=dataloder(root=root,batchsize=batches,imagesize=imagesize,ratio=ratio)
    
    model = UnetPlusPlus(input_channel=3, num_classes=1, deep_supervision=deep_supervision)
    model.to(device)
    
    start_time = time.time()
    TRAIN_MIOU,TRAIN_BIOU,TRAIN_LOSS,TEST_MIOU,TEST_BIOU,TEST_LOSS=train(model,Train_data,Test_data,epoches,lr,deep_supervision=deep_supervision)
    duration=time.time()-start_time
    print('train_total_time=%.2f' %duration)
    
    draw(TRAIN_MIOU,TRAIN_BIOU,TRAIN_LOSS,TEST_MIOU,TEST_BIOU,TEST_LOSS,epoches,png_root)