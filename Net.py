import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# VGGBlock
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x
 
class UnetPlusPlus(nn.Module):
    def __init__(self, input_channel,num_classes,deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.in_channel = input_channel
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        self.channel = [64, 128, 256, 512, 1024]
        
        self.CONV3_1 = VGGBlock(512*2, 512)
 
        self.CONV2_2 = VGGBlock(256*3, 256)
        self.CONV2_1 = VGGBlock(256*2, 256)
 
        self.CONV1_1 = VGGBlock(128*2, 128)
        self.CONV1_2 = VGGBlock(128*3, 128)
        self.CONV1_3 = VGGBlock(128*4, 128)
 
        self.CONV0_1 = VGGBlock(64*2, 64)
        self.CONV0_2 = VGGBlock(64*3, 64)
        self.CONV0_3 = VGGBlock(64*4, 64)
        self.CONV0_4 = VGGBlock(64*5, 64)
 
 
        self.stage_0 = VGGBlock(self.in_channel, 64)
        self.stage_1 = VGGBlock(64, 128)
        self.stage_2 = VGGBlock(128, 256)
        self.stage_3 = VGGBlock(256, 512)
        self.stage_4 = VGGBlock(512, 1024)
 
        self.pool = nn.MaxPool2d(2)
    
        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
 
        
        # 分割头
        self.final_super_0_1 = nn.Sequential(
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_2 = nn.Sequential(
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_3 = nn.Sequential(
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_4 = nn.Sequential(
          nn.Conv2d(64, self.num_classes, 3, padding=1),
        )        
 
        
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        x_0_1 =  self.CONV0_1(torch.cat([self.upsample_0_1(x_1_0) , x_0_0], 1))
        
        x_1_1 = self.CONV1_1(torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1))
        
        x_2_1 = self.CONV2_1(torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1))
        
        x_3_1 = self.CONV3_1(torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1))
 
        x_2_2 = self.CONV2_2(torch.cat([self.upsample_2_2(x_3_0), x_2_0, x_2_1], 1))
        
        x_1_2 = self.CONV1_2(torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1))
        
        x_1_3 = self.CONV1_3(torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1))
 
        x_0_2 = self.CONV0_2(torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1))
        
        x_0_3 = self.CONV0_3(torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1))
        
        x_0_4 = self.CONV0_4(torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1))
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)
 
 
if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 80, 80)).to(device)
    model = UnetPlusPlus(input_channel=3,num_classes=1, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    print(outputs.shape)    
    
    print("deep_supervision: True")
    deep_supervision = True
    model = UnetPlusPlus(input_channel=3,num_classes=1, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
      print(out.shape)
 
 