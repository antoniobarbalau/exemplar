from classifier import VGG
from optimizer import ESMomentum
from torch import nn
from vae import VAE
import numpy as np
import torch
import time

np.random.seed(0)

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

generator = VAE().to(device)
generator.load_state_dict(torch.load('./vae_state_dict'))
generator.train(False)

classifier = VGG().to(device)
classifier.load_state_dict(torch.load('./classifier_state_dict'))
classifier.train(False)

def optimize(label):
    target_softmax = np.eye(7)[label]
    target_softmax = torch.tensor([target_softmax]).to(device)
    start_time = time.time()
    encoding = torch.tensor(
        np.random.uniform(-5., 5., size = (1, 256)),
        dtype = torch.float32,
        requires_grad = True
    ).to(device)

    image = generator.generate(encoding)
    softmax = torch.softmax(classifier(image), -1)
    
    loss = torch.pow(softmax - target_softmax, 2).sum()
    grad = torch.autograd.grad(loss, encoding)[0]
    exemplar_confidence = 0.
    n_iterations = 0
    while exemplar_confidence < .95 and n_iterations < 5000:
        n_iterations += 1
        image = generator.generate(encoding)
        softmax = torch.softmax(classifier(image), -1)
        
        loss = torch.pow(softmax - target_softmax, 2).sum()
        exemplar_confidence = softmax[0][label]
        grad = .7 * grad + .3 * torch.autograd.grad(loss, encoding)[0]
        encoding = encoding - 1 * grad
    return image, n_iterations, time.time() - start_time, exemplar_confidence

n_tests = 1000
iterations = []
durations = []
for test_n in range(n_tests):
    label = np.random.randint(7)
    print(f'Synthesizing exemplar: {test_n} / {n_tests}   ', end = '\r')
    exemplar, n_iterations, duration, confidence = optimize(
        label = label,
    )
    if n_iterations == 5000:
        print()
        print(
            f'FAILED: Sythesizing exemplar {test_n} with label: {label}. ' +
            f'Synthesized sample confidence: {confidence}'
        )
    iterations.append(n_iterations)
    durations.append(duration)
iterations = np.array(iterations)
durations = np.array(durations)
print()
print(f'Number of failed synthetizations: {np.sum(iterations == 5000)}')
print(
    f'Average number of forward passes when convergence was reached: ' +
    f'{np.mean(iterations[iterations != 5000])}'
)
print(
    f'Average synthetization duration when ' +
    f'convergence was reached: {np.mean(durations[iterations != 5000])}'
)



