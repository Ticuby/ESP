import torch.nn as nn
from Net import resnet
import torch
from .BERT.self_attention import self_attention
import torchvision

class ResNet50(nn.Module):
    def __init__(self,num_classes=4,BERT=False):
        super(ResNet50,self).__init__()
        self.resnet = resnet.resnet101(pretrained=True)
        for i in self.resnet.parameters():
            i.requires_grad = False
        self.fc = nn.AvgPool2d
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if BERT:
            self.self_attention = self_attention(self.hidden_size, 1, hidden=self.hidden_size,
                                                 n_layers=self.n_layers, attn_heads=self.attn_heads)
        self.fc = nn.Linear(512 * resnet.Bottleneck.expansion, num_classes)
    def forward(self,x):
        x = self.resnet(x)  #b,3,224,224 -> b,2048,7,7
        x = self.avgpool(x) #            -> b,2048,1,1
        x = torch.flatten(x, 1)#         -> b,2048
        x = self.fc(x)#                  -> b,4
        return x

