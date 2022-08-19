import torch
from torchvision import transforms
from Unetplusplus import Unet_plus_plus
from Net import UnetPlusPlus
from dataset import dataloder
from PIL import Image

def predict(Test_data,root,deep_supervision):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number=0
    model=UnetPlusPlus(input_channel=3, num_classes=1, deep_supervision=deep_supervision).to(device)
    model.load_state_dict(torch.load('./model/Unet_Plus_Plus_model.pth'))
    with torch.no_grad():
        for test_image_batch, test_mask_batch in Test_data:
            test_image_batch = test_image_batch.to(device)
            test_mask_batch = test_mask_batch.to(device)
            number += 1
            B, H, W = test_mask_batch.shape
            for i in range(B):
                pic1 = transforms.ToPILImage()(test_image_batch[i])
                pic2 = transforms.ToPILImage()(test_mask_batch[i])

                pic1.save('%s/%d_origin.png' % (root, number))
                pic2.save('%s/%d_mask.png' % (root, number))
                
            outputs = model(test_image_batch)
            outs = outputs[-1].squeeze(dim=1)
            
            pre_outs = torch.sigmoid(outs).data.cpu().numpy()
            pre_outs[pre_outs > 0.5] = 255          #前景
            pre_outs[pre_outs <= 0.5] = 0           #背景
            
            for j in range(B):
                pre_out = pre_outs[j].astype('uint8')
                preout = Image.fromarray(pre_out)
                preout.save('%s/%d_predict.png' % (root, number))
                
                
if __name__ == '__main__':
    
    root='./archive/weizmann_horse_db'
    ratio=0.15
    batches=1
    imagesize=80
    deep_supervision=True
    Train_data,Test_data=dataloder(root=root,batchsize=batches,imagesize=imagesize,ratio=ratio)
    
    predict(Test_data,'./predict/',deep_supervision)
    
