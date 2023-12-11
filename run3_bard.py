import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as data
import torchvision
import random

from torchvision.utils import save_image

class CouplingLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels * 2, 1)
        )

    def forward(self, x, cond):
        # print(x.shape)
        # print(cond.shape)
        # print("\n\n\n")
        h = torch.cat([x], dim=1)
        # h = torch.cat([x, cond], dim=1)
        s, t = torch.split(self.net(h), x.shape[1], dim=1)
        return x * torch.exp(s) + t, torch.abs(s)

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_coupling_layers):
        super().__init__()
        self.coupling_layers = nn.ModuleList([CouplingLayer(in_channels, hidden_channels).to("cuda:0") for _ in range(num_coupling_layers)])

    def forward(self, x, cond):
        log_det_jacobian = 0.0
        for layer in self.coupling_layers:
            # print("CNF")
            # print(x.shape)
            # print(cond.shape)
            # print("\n\n\n")
            x, log_det = layer(x, cond)
            log_det_jacobian += log_det
        return x, log_det_jacobian

class CNFImageToImageTranslator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_coupling_layers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.cnf = ConditionalNormalizingFlow(hidden_channels * 2, hidden_channels, num_coupling_layers).to("cuda:0")
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1),
        )

    def forward(self, x, cond):
        z = self.encoder(x)
        cond = self.encoder(cond)
        z, log_det_jacobian = self.cnf(z, cond)
        reconstruction = self.decoder(z)
        return reconstruction, log_det_jacobian


# Example usage
cnf = CNFImageToImageTranslator(3, 3, 64, 4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

cnf.to(device)

x = torch.rand(1, 3, 64, 64).to("cuda:0")
cond = torch.rand(1, 3, 64, 64).to("cuda:0")
reconstruction, log_det_jacobian = cnf(x, cond)

optimizer = optim.Adam(cnf.parameters(), lr = 0.001)
# optimizer = optim.RMSprop(cnf.parameters(),lr=0.0001)
# optimizer = optim.Adagrad(cnf.parameters(),lr=0.0001)

loss_criterion = nn.MSELoss()


img_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.0001, 0.0001,0.9), (0.229, 0.224, 0.225))
    torchvision.transforms.Normalize((0.9,0.0001,0.0001), (0.081, 0.222, 0.2201)),
    # torchvision.transforms.Grayscale(num_output_channels=3)
    torchvision.transforms.ConvertImageDtype(dtype= torch.float32)
    ])

batch_size = 8
img_data = torchvision.datasets.ImageFolder(root="input_images2",transform=img_transform)
data_loader = data.DataLoader(img_data, batch_size, shuffle=False)

target_img_data = torchvision.datasets.ImageFolder(root="output_images2",transform=img_transform)
target_data_loader = data.DataLoader(target_img_data, batch_size, shuffle=False)

print("ENDED")


num_epochs = 2000
interval = 9
targets = list(target_data_loader)
for epoch in range(num_epochs):

    for i, v in enumerate(data_loader):
        target_image, _ = targets[i]
        # target_image = target_images[i].to("cuda:0")
        # Extract source image and conditioning variable
        # print("i= ",i)
        # print(v[0].shape)
        x = v[0].to("cuda:0")

        cond = torch.zeros(1, 3, 64, 64).to("cuda:0")        # im1 = Image.fromarray(target_image.numpy()[0][0])
        # print(im1.size)
        # im1.save("target_image.jpg")


        # Generate reconstructed image
        reconstruction, log_det_jacobian = cnf(x, cond)

        # Calculate loss
        # loss = reconstruction_loss(reconstruction, target_image) + adversarial_loss(reconstruction)
        # print("target", target_image.shape)
        # print("reconstruction ", reconstruction.shape)


        try:
            # loss = 0
            # for rec in reconstruction.to("cuda:0"):
                # loss += loss_criterion(rec, target_image)
            loss = loss_criterion(reconstruction.to("cuda:0"), target_image.to("cuda:0"))
            # loss = loss_criterion(reconstruction.to("cuda:0"), target_image)
            # loss = torch.sum(log_det_jacobian)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if random.random()<0.2:

        
        except:
            print("SKIPPED!!!!!!!!!!")

        
        # Print training progress
        # if i % print_interval == 0:
    save_image(reconstruction,"reconstructed_image.png")
    save_image(target_image,"target_image.png")
    print(f'Epoch: {epoch}/{num_epochs}, Batch: {i}/{len(data_loader)}, Loss: {loss.item()}')

    # Evaluate model on validation set
    
    # save_image(target_image,"target_image.png")
    # save_image(reconstruction,"reconstructed_image.png")

    torch.jit.save(torch.jit.trace(cnf, (torch.randn(1, 3, 64, 64).to("cuda:0"),torch.randn(1, 3, 64, 64).to("cuda:0"))), "saved_model.pt")
print("DONE!!")