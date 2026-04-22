import os
import cv2

import numpy as np
import torch
import torchsummary
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    sample: int = 25600
    batch_size: int = 16
    channel: int = 1
    im_size: int = 128
    epochs: int = 100
    
    block = int(sample / batch_size)
    path = ""
    
    image_files = os.listdir(path)
    print(len(image_files))
    
    images = np.empty((sample, im_size, im_size, 3))
    for k, file in enumerate(image_files):
        if k == sample:
            break
        images[k] = cv2.imread(path+"/"+file)
    
    print(images.shape)
    images /= 255.0
    images = images.transpose(0, 3, 1, 2)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = model.AutoEncoder().to(device)
    net.apply(weights_init)
    
    #model = torch.load('model.pt', weights_only=False,
    #                   map_location=torch.device("cpu"))
    
    torchsummary.summary(net, (channel, im_size, im_size))
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999))
    
    im = Image.new("RGB", (im_size, im_size))
    
    for t in range(epochs):
        out_mean = 0
        for k in range(block):
            train = images[k*batch_size:(k+1)*batch_size, 0]
            train = train.reshape(batch_size, 1, im_size, -1)
            train = np.float32(train)
            data = torch.from_numpy(train).clone()
    
            real_image = data.to(device)
            sample_size = real_image.size(0)
           
            net.zero_grad()
            output = net(real_image)
            
            loss = criterion(output, real_image)
            out_mean += output.mean().item()
            loss.backward()
            optimizer.step()
        
        for j in range(im_size):
           for i in range(im_size):
               rgb = int(255*(output[0, 0, j, i])+0.5)
               im.putpixel((i, j), (rgb, rgb, rgb))
        im.save("fake/"+str(t)+".png")
    
        print(out_mean)
    
    torch.save(model, 'model.pt')
    
