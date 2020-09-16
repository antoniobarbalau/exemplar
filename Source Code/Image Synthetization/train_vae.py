from torchvision import transforms
import cv2
import numpy as np
import torch
from vae import VAE
import torchvision

dataset = torchvision.datasets.ImageFolder(
    root = './dataset/fer/train',
    transform = transforms.Compose([
        transforms.Resize(48),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor(),
        transforms.Normalize(
            (.5,), (.5,)
        ),
    ]),
)

vae = VAE().cuda()
optimizer = torch.optim.Adam(vae.parameters(), lr = 1e-3)

n_epochs = 120
scheduler = [
    0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 1., 1., 1., 1.
] * 8

for epoch in range(n_epochs):
    print(epoch, end = '\r')
    beta = scheduler[epoch]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 64,
        shuffle = True,
        num_workers = 2,
        drop_last = True
    )
    for iter_n, batch in enumerate(dataloader):
        images = batch[0].cuda()
        
        vae.zero_grad()
        means, stds, embeddings, reconstructions = vae(images)
        reconstruction_loss = torch.pow(images - reconstructions, 2).sum(
            axis = [1, 2, 3]
        )
        kl_loss = - 0.5 * torch.sum(
            1 + stds - torch.pow(means, 2) - torch.exp(stds),
            axis = -1
        )
        loss = torch.mean(reconstruction_loss + beta * kl_loss)
        loss.backward()
        optimizer.step()

torch.save(vae.state_dict(), f'./vae_state_dict')
