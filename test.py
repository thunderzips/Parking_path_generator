import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as data
import torchvision
import time
from torchvision.utils import save_image

img_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Grayscale(num_output_channels=3)
    # torchvision.transforms.Normalize((0.9,0.0001,0.0001), (0.081, 0.222, 0.2201)),
    torchvision.transforms.ConvertImageDtype(dtype= torch.float32)
    ])

batch_size = 16
img_data = torchvision.datasets.ImageFolder(root="input_images2",transform=img_transform)
data_loader = data.DataLoader(img_data, batch_size, shuffle=False)

target_img_data = torchvision.datasets.ImageFolder(root="output_images2",transform=img_transform)
target_data_loader = data.DataLoader(target_img_data, batch_size, shuffle=False)




cnf = torch.jit.load("model_1.pt")



cond = torch.zeros(1, 3, 64, 64).to("cuda:0")

targets = list(target_data_loader)

for i, v in enumerate(data_loader):
    reconstruction, log_det_jacobian = cnf(v[0].to("cuda:0"), cond)
    target_images, _ = targets[i]

    save_image(target_images,"target_image.png")
    save_image(reconstruction,"reconstructed_image.png")
    print(i)
    time.sleep(2)