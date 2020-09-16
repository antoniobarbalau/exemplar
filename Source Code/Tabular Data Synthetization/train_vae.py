import numpy as np
import pandas as pd
import torch
from dataset import Adult
from vae import VAE

device = torch.device('cpu')
ds = Adult()
n_samples = len(ds.dataset)
vae = VAE().to(device)
optim = torch.optim.Adam(vae.parameters())

test_indexes = np.random.randint(0, n_samples, size = (1,))
test_numerical, test_linear, test_categorical, test_labels = ds.preprocess(
    ds.dataset.take(test_indexes)
)
test_numerical = torch.Tensor(test_numerical).to(device)
test_linear = torch.Tensor(test_linear).to(device)
test_categorical = torch.LongTensor(test_categorical).to(device)
def test():
    vae.train(False)
    output = vae(test_numerical, test_linear, test_categorical)
    numerical = np.array(test_numerical.detach().cpu()).ravel()
    out_numerical = np.array(output[0].detach().cpu()).ravel()
    for pair in list(zip(numerical, out_numerical)):
        print(pair)
    linear = np.array(test_linear.detach().cpu()).ravel()
    out_linear = np.array(output[1].detach().cpu()).ravel()
    for pair in list(zip(linear, out_linear)):
        print(pair)
    categorical = np.array(test_categorical.detach().cpu()).ravel()
    for i in range(8):
        categorical = np.array(test_categorical[:, i].detach().cpu()).ravel()
        out_categorical = np.array(output[2][i].argmax(-1).detach().cpu()).ravel()
        for pair in list(zip(categorical, out_categorical)):
            print(pair)
    vae.train(True)


n_epochs = 30
batch_size = 64
n_iterations = n_samples // batch_size
cce = torch.nn.CrossEntropyLoss()
for epoch_n in range(n_epochs):
    print(f'Epoch {epoch_n} / {n_epochs}')
    for iter_n in range(n_iterations):
        vae.zero_grad()

        indexes = np.random.randint(0, n_samples, size = (batch_size,))
        numerical, linear, categorical, labels = ds.preprocess(
            ds.dataset.take(indexes)
        )
        numerical = torch.Tensor(numerical).to(device)
        linear = torch.Tensor(linear).to(device)
        categorical = torch.LongTensor(categorical).to(device)
        output = vae(numerical, linear, categorical, sample = True)

        loss = (output[0] - numerical).pow(2).sum()
        loss += torch.sum(torch.stack([
            cce(output[2][i], categorical[:, i])
            for i in range(8)
        ]))
        means, stds = output[-2], output[-1]
        kl_loss = - 0.5 * torch.sum(
            1 + stds - torch.pow(means, 2) - torch.exp(stds)
        )
        loss += .3 * kl_loss

        loss.backward()
        optim.step()

    test()

torch.save(vae.state_dict(), f'./adult_vae')


