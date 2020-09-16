import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from classifier import VGG
import torchvision
from torchvision import transforms
from torch.autograd import Variable


device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
loss_function = torch.nn.CrossEntropyLoss()

dataset = torchvision.datasets.ImageFolder(
    root = './dataset/fer/train/',
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

dataset_test =  torchvision.datasets.ImageFolder(
    root = './dataset/fer/test/',
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
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

lr = 0.01
cl = VGG().to(device)
optimizer = torch.optim.SGD(cl.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
learning_rate_decay_start = 17
learning_rate_decay_every = 1
learning_rate_decay_rate = .9

n_epochs = 50
for epoch in range(n_epochs):
    cl.train(True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 64,
        shuffle = True,
        num_workers = 2,
        drop_last = True
    )
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = lr * decay_factor
        set_lr(optimizer, current_lr)
    else:
        current_lr = lr
    for iter_n, batch in enumerate(dataloader):

        images = batch[0].to(device)
        targets = batch[1].to(device)
        
        cl.zero_grad()
        outputs = cl(images)
        loss = loss_function(outputs, targets)
        acc = outputs.max(1)[1].eq(targets).float().mean()
        acc = acc.detach().cpu()
        print(f'{epoch}, {iter_n}, {acc}', end = '\r')

        loss.backward()
        clip_gradient(optimizer, 0.1)
        optimizer.step()

    cl.train(False)
    dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size = 16,
        num_workers = 2,
        drop_last = True
    )
    accs = []
    for iter_n, batch in enumerate(dataloader):
        images = batch[0].to(device)
        targets = batch[1].to(device)
        
        with torch.no_grad():
            outputs = cl(images)
            acc = outputs.max(1)[1].eq(targets).float().mean()
            acc = acc.detach().cpu()
            accs.append(acc)
    print(f'{epoch}, {np.mean(accs)}                        ')
    torch.save(cl.state_dict(), f'./classifier_state_dict')
