import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as data
import torchvision
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import copy
import cv2

from helpers import TextFileDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

img_size = 64
batch_size = 1

loss_criterion = nn.MSELoss() #nn.L1Loss() #

def get_loss(tar_img, generated):
    '''
    TODO: Model the spline curve in terms of the generated points
    '''
    return loss_criterion(tar_img,generated)

img_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Grayscale(num_output_channels=1),
    # torchvision.transforms.ConvertImageDtype(dtype= torch.float32)
    ])

class TestNet(nn.Module):
    def __init__(self, img_size, input_channels, out_channels, hidden_layers):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.inp_layer = nn.Sequential(
            nn.Linear(input_channels*img_size*img_size,input_channels*img_size*img_size//2),
            nn.ReLU(inplace=True)
        )
        self.hidden = nn.Sequential(
            nn.Linear(input_channels*img_size*img_size//2,input_channels*img_size*img_size//2),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(input_channels*img_size*img_size//2,input_channels*img_size*img_size//2),
            nn.ReLU(inplace=True)
        )
        self.op_layer = nn.Sequential(
            nn.Linear(input_channels*img_size*img_size//2, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.inp_layer(x)
        for _ in range(self.hidden_layers):
            x = self.hidden(x)
        x = self.l2(x)
        x = self.op_layer(x)
        return x

cnf = TestNet(img_size, 3, 40, 1)
cnf.to(device)

optimizer = optim.Adam(cnf.parameters(), lr = 0.001)

img_data = torchvision.datasets.ImageFolder(root="input_images",transform=img_transform)
data_loader = data.DataLoader(img_data, batch_size, shuffle=False)

target_points_data = TextFileDataset(root_dir="output_points")
target_data_loader = data.DataLoader(target_points_data, batch_size, shuffle=False)

targets = list(target_data_loader)

j = 20
num_epochs = 100
for epoch in range(num_epochs):
    for i, v in enumerate(data_loader):
        target_image, _ = targets[i]
        target_image.to(device)
        x = v[0].to(device)
        x = torch.reshape(x,(-1,3*img_size*img_size))

        reconstruction = cnf(x)
        reconstruction = reconstruction.to(device)
        # if (j+i+1)%30 == 1:
        #     rec = reconstruction
        #     tar = target_image
        #     ori = v[0]

        processed_image = copy.copy(x)
        processed_image = processed_image.detach().cpu().numpy()
        processed_image = cv2.resize(processed_image, (img_size, img_size))
        cv2.imwrite("saved_input.png",processed_image)

        try:
            loss = get_loss(target_image.to(device), reconstruction.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            print("SKIPPED!!!!!!!!!!")
        

    if epoch%10 == 0:
        torch.jit.save(torch.jit.trace(cnf, (torch.randn(batch_size,3*img_size*img_size).to(device))), "saved_model.pt")
    
    j += 1
    
    print(f'Epoch: {epoch}/{num_epochs}, Batch: {i}/{len(data_loader)}, Loss: {loss.item()}')#, LogSum: {log_sum}')

print("DONE!!")