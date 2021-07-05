import json
from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np

# Load ViT
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=False, num_classes=4)
model.train()

# Load image
# NOTE: Assumes an image `img.jpg` exists in the current directory
# img = transforms.Compose([
#     transforms.Resize((384, 384)),     transforms.ToTensor(),
#     transforms.Normalize(0.5, 0.5),])(Image.open('img.jpg')).unsqueeze(0)
# print(img.shape) # torch.Size([1, 3, 384, 384])
img = np.ones([2,3,384,384],dtype=np.float32)
img = torch.from_numpy(img)

# Classify
with torch.no_grad():
    outputs = model(img)
print(outputs.shape)  # (1, 1000)
