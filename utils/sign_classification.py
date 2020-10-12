import torch
import torch.nn as nn


class ConvBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, factor = 2):

    super().__init__()

    #self.expand = nn.Conv2d(in_channels, factor * out_channels, kernel_size=1)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    #self.squzee = nn.Conv2d(factor * out_channels, out_channels, kernel_size=1)
    self.relu = nn.ReLU()

  def forward(self, t):
    #t = self.relu(self.expand(t))
    t = self.relu(self.bn(self.conv(t)))
    #t = self.relu(self.squzee(t))

    return t

x = torch.randn(1, 3, 28, 28)
conv1 = ConvBlock(3, 16, 3)
out = conv1(x)
conv2 = nn.Conv2d(3, 16, 3)
out = conv2(x)
out.shape

class SignNet(nn.Module):

  def __init__(self):

    super().__init__()
    #ConvBlock = nn.Conv2d

    self.conv1 = ConvBlock(3, 16, 3, padding=1)
    
    self.conv2 = ConvBlock(16, 32, 3, stride=1, padding=1)

    self.conv3 = ConvBlock(32, 32, 3, stride=2, padding=1)
    
    self.conv4 = ConvBlock(32, 32, 5, padding=2)
    
    self.conv5 = ConvBlock(32, 64, 5, stride=2, padding=2)

    self.conv6 = ConvBlock(64, 64, 5, stride=1, padding=2)
    
    self.conv7 = ConvBlock(64, 128, 5)
    
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.drop1 = nn.Dropout(0.5)
    self.drop2 = nn.Dropout(0.2)
    self.linear2 = nn.Linear(128, 43)
    self.relu = nn.ReLU()
  
  def forward(self, t):

    #print(t.shape)
    t = self.relu(self.conv1(t))
    #print(t.shape)
    t = self.relu(self.conv2(t))
    #print(t.shape)
    t = self.relu(self.conv3(t))
    #print(t.shape)
    t = self.relu(self.conv4(t))
    #print(t.shape)
    t = self.relu(self.conv5(t))
    #print(t.shape)
    t = self.relu(self.conv6(t))
    #print(t.shape)
    t = self.relu(self.conv7(t))
    #print(t.shape)
    t = self.avgpool(t)
    #print(t.shape)
    t = t.view(t.shape[0], -1)
    #print(t.shape)
    t= self.drop1(t)
    return self.linear2(t)