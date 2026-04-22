import os
import cv2

import numpy as np
import torch
import torchsummary
from PIL import Image

if __name__ == '__main__':
    sample: int = 80
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

    net = torch.load('model.pt', weights_only=False)

    torchsummary.summary(net, (channel, im_size, im_size))

    train = images[0:batch_size, 0]
    train = train.reshape(batch_size, 1, im_size, -1)
    train = np.float32(train)
    
    for b in range(batch_size):
        for j in range(10):
            for i in range(10):
                train[b, 0, j+64, i+64] = 0.0
    
    data = torch.from_numpy(train).clone()

    real_image = data.to(device)
    sample_size = real_image.size(0)
    # print(real_image.shape)

    output = net(real_image)
    #

    im = Image.new("RGB", (im_size, im_size))
    
    for b in range(batch_size):
        for j in range(im_size):
            for i in range(im_size):
                rgb = int(255*(output[b, 0, j, i])+0.5)
                im.putpixel((i, j), (rgb, rgb, rgb))
        im.save("val/"+str(b)+".png")
