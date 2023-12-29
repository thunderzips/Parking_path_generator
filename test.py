import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as data
import torchvision
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import sys
import cv2
import time

from helpers import TextFileDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = 64
batch_size = 1

img_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ConvertImageDtype(dtype= torch.float32)
    ])


cnf = torch.jit.load("model_40points_1channel.pt").to(device)



img_data = torchvision.datasets.ImageFolder(root="input_images",transform=img_transform)
data_loader = data.DataLoader(img_data, batch_size, shuffle=False)

# target_points_data = TextFileDataset(root_dir="output_points")
# target_data_loader = data.DataLoader(target_points_data, batch_size, shuffle=False)

# targets = list(target_data_loader)

num_epochs = 1000
for epoch in range(num_epochs):
    for i, v in enumerate(data_loader):
        # target_image, _ = targets[i]
        # target_image.to(device)
        x = v[0].to(device)
        x = torch.reshape(x,(-1,img_size*img_size))
        # x = torch.reshape(x,(-1,3*img_size*img_size))

        reconstruction = cnf(x)
        # reconstruction = reconstruction.to(device)
        # print(reconstruction)

        processed_image = x.detach().cpu().numpy()
        reconstruction = reconstruction.detach().cpu().numpy()

        processed_output = np.zeros((100,100,3),dtype=np.uint8)
        processed_output.fill(255)

        reconstruction = np.reshape(reconstruction,(20,2))

        for pix in reconstruction:
            pix_ = (int(pix[0]),int(pix[1]))
            cv2.circle(processed_output, pix_, 1, (0,0,255), 1)

        processed_image = cv2.resize(processed_image, (img_size, img_size))
        # processed_output = cv2.resize(processed_output, (img_size, img_size))

        cv2.imwrite("saved_input.jpg",processed_image)
        cv2.imwrite("saved_output.jpg",processed_output)

        print(i)

        time.sleep(1)
        # sys.exit()

print("DONE!!")