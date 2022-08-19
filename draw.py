import matplotlib.pyplot as plt
import numpy as np
import torch

def draw(TRAIN_MIOU,TRAIN_BIOU,TRAIN_LOSS,TEST_MIOU,TEST_BIOU,TEST_LOSS,epoches,root):
    epoch=np.arange(1,epoches+1)
    TRAIN_LOSS=torch.tensor(TRAIN_LOSS, device='cpu')
    TEST_LOSS=torch.tensor(TEST_LOSS, device='cpu')
    
    plt.figure(figsize=(8, 8))
    plt.title('mIoU')
    plt.xlabel('epoch')
    plt.ylabel('mIoU')
    plt.plot(epoch, TRAIN_MIOU, label='Train')
    plt.plot(epoch, TEST_MIOU, label='Test')
    plt.legend(loc='upper left')
    plt.savefig('%s/mIoU.png'%(root))
    plt.clf()
    
    plt.figure(figsize=(8, 8))
    plt.title('Boundary IoU')
    plt.xlabel('epoch')
    plt.ylabel('Boundary IoU')
    plt.plot(epoch, TRAIN_BIOU, label='Train')
    plt.plot(epoch, TEST_BIOU, label='Test')
    plt.legend(loc='upper left')
    plt.savefig('%s/BIoU.png'%(root))
    plt.clf()
    
    plt.figure(figsize=(8, 8))
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.plot(epoch, TRAIN_LOSS, label='Train')
    plt.plot(epoch, TEST_LOSS, label='Test')
    plt.legend(loc='upper left')
    plt.savefig('%s/Loss.png'%(root))
    plt.clf()

    plt.close()

