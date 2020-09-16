from classifier import VGG
from torch import nn
from vae import VAE
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F

np.random.seed(0)

device = torch.device('cuda:0')

generator = VAE().cuda()
generator.load_state_dict(torch.load('./vae_state_dict'))
generator.train(False)

classifier = VGG().cuda()
classifier.load_state_dict(torch.load('./classifier_state_dict'))
classifier.train(False)

class Specimen(object):
    def __init__(
        self,
        position = None,
        momentum = None
    ):
        self.position = position
        if position is None:
            self.position = np.random.uniform(-5., 5., size = (256,))
        self.momentum = momentum

    def mutate(self):
        alpha = .3
        velocity = np.random.normal(scale = .5, size = (256,))
        if self.momentum is not None:
            velocity = alpha * self.momentum + (1 - alpha) * velocity
        return (
            self.position + velocity,
            velocity
        )

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
    while exemplar_confidence < .95 and n_iterations < 500:
        n_iterations += 1
        image = generator.generate(encoding)
        softmax = torch.softmax(classifier(image), -1)
        
        loss = torch.pow(softmax - target_softmax, 2).sum()
        exemplar_confidence = softmax[0][label]
        grad = .7 * grad + .3 * torch.autograd.grad(loss, encoding)[0]
        encoding = encoding - 1 * grad

        image = (image - image.min()) / (image.max() - image.min())
        image = np.array(image.detach().cpu())
        exemplar = np.ones((4 * 48, 4 * 48))
        exemplar[
            72: 72 + 48,
            72: 72 + 48,
        ] = image
        exemplar_confidence = exemplar_confidence.detach().cpu().item()
        exemplar = np.uint8(np.tile(np.expand_dims(exemplar, axis = -1), [1, 1, 3]) * 255.)
        cv2.imshow('Exemplar Synthetization', exemplar)
        cv2.waitKey(10)
    return image, n_iterations, time.time() - start_time, exemplar_confidence

n_tests = 10
label = 5 if len(sys.argv) == 1 else int(sys.argv[1])
for test_n in range(n_tests):
    image, n_iterations, duration, exemplar_confidence = optimize(
        label = label,
    )

    exemplar = np.ones((4 * 48, 4 * 48))
    exemplar[
        72: 72 + 48,
        72: 72 + 48,
    ] = image
    cv2.putText(
        exemplar,
        f'Confidence: {str(exemplar_confidence * 100)[:4]}%',
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    exemplar = np.uint8(np.tile(np.expand_dims(exemplar, axis = -1), [1, 1, 3]) * 255.)
    cv2.imshow('Exemplar Synthetization', exemplar)
    cv2.waitKey(2000)


