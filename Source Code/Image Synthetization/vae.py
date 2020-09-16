import torch
from generator import Generator
from discriminator import Discriminator
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.generator = Generator()
        self.encoder = Discriminator()

    def sample(self, encoder_output):
        means = encoder_output[:, :256]
        stds = encoder_output[:, 256:]
        return (
            means,
            stds,
            means + torch.exp(.5 * stds) * torch.randn(
                size = (encoder_output.size(0), 256),
                device = 'cuda'
            )
        )

    def generate(self, noise):
        return self.generator(noise)

    def forward(self, inputs):
        means, stds, encodings = self.sample(self.encoder(inputs))
        reconstructions = self.generator(encodings)

        return means, stds, encodings, reconstructions

