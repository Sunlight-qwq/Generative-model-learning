import torch
from denoising_diffusion_pytorch import Unet
from torchvision.utils import save_image


shape = (1, 28, 28)
num_steps = 1000
gpu = torch.device("cuda")

betas = torch.linspace(0.0001, 0.02, num_steps).to(gpu)
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_sqrt = alphas_bar.sqrt()
betas_bar_sqrt = torch.sqrt(1. - alphas_bar)

unet = Unet(dim=32, dim_mults=(1, 2, 4), channels=1)
unet = torch.load("./diffusion/unet.pkl")

@torch.no_grad()
def generate(num: int) -> torch.Tensor:
    unet.eval()
    x_t = torch.randn((num, *shape)).to(gpu)
    for t in reversed(range(num_steps)):
        z_t = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        t = t * torch.ones(num).long().to(gpu)
        e_hat = unet(x_t, t)
        t = t.reshape(-1, 1, 1, 1)
        x_t = 1 / alphas[t].sqrt() * (x_t - betas[t] / betas_bar_sqrt[t] * e_hat) \
            + betas[t] * z_t
    return x_t


if __name__ == "__main__":
    from torchvision.utils import save_image
    NUM = 10
    imgs = generate(NUM)
    for k in range(NUM):
        save_image(imgs[k], f"./diffusion/gen_imgs/{k + 1}.png")
