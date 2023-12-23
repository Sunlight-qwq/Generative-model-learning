import torch, torchvision
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from denoising_diffusion_pytorch import Unet
from typing import *

mnist_dataset = torchvision.datasets.MNIST(root='./datasets', 
    train=True, transform=torchvision.transforms.ToTensor(),
    download=True
)

shape = (1, 28, 28)
num_steps = 1000
gpu = torch.device("cuda")

betas = torch.linspace(0.0001, 0.02, num_steps).to(gpu)
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_sqrt = alphas_bar.sqrt()
betas_bar_sqrt = torch.sqrt(1. - alphas_bar)

def get_x_t(x_0: torch.Tensor, t: torch.Tensor, e_t: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Get x_t by x_0 and t.
    """
    if e_t is None:
        e_t = torch.randn_like(x_0)
    
    return x_0 * alphas_bar_sqrt[t].reshape(-1, 1, 1, 1)\
         + e_t * betas_bar_sqrt[t].reshape(-1, 1, 1, 1)

unet = Unet(dim=8, dim_mults=(1, 2, 4), channels=1)

batch_size = 4096
epochs = 50
lr = 1e-4

loader = data.DataLoader(mnist_dataset, batch_size, shuffle=True)
mse = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr)

unet.to(gpu)
unet.train()

for epoch in range(epochs):
    sum_loss = 0.
    cnt = 0

    for x, _ in loader:
        x = x.to(gpu)
        t = torch.randint(0, num_steps, size=(x.shape[0],)).long().to(gpu)
        e_t = torch.randn_like(x)
        x_t = get_x_t(x, t, e_t)
        e_hat: torch.Tensor = unet(x_t, t)
        loss: torch.Tensor = mse(e_hat, e_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sum_loss += float(loss)
        cnt += 1
    
    print(f"Epoch {epoch + 1}, loss {sum_loss / cnt}")

torch.save(unet, "./unet.pkl")
