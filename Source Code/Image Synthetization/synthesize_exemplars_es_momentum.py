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

def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    return np.sum(np.power(softmax - np.eye(7)[label], 2))

def plot(images, wait = 250):
    images = (images - images.min()) / (images.max() - images.min())
    images = images.detach().cpu()
    images = np.reshape(images, (4, 4, 48, 48))
    images = np.hstack(images)
    images = np.hstack(images)
    images = np.expand_dims(images, axis = -1)
    images = np.tile(images, [1, 1, 3])
    images = np.uint8(images * 255.)
    cv2.imshow('Exemplar Synthetization', images)
    cv2.waitKey(wait)

n_tests = 10
label = 5 if len(sys.argv) == 1 else int(sys.argv[1])
for test_n in range(n_tests):
    specimens = [
        Specimen() for _ in range(50)
    ]
    exemplar_confidence = 0.
    n_iterations = 0
    while exemplar_confidence < .97:
        n_iterations += 1
        encodings = np.array([
            s.position for s in specimens
        ], dtype = np.float32)
        images = generator.generate(
            torch.Tensor(encodings).cuda()
        )
        plot(images[:16])
        with torch.no_grad():
            softmaxes = classifier(images).detach().cpu()
        losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
        specimens = [
            specimens[index]
            for index in np.argsort(losses)[:10]
        ]
        specimens = specimens + [
            Specimen(*s.mutate()) for s in specimens
        ] + [
            Specimen(*s.mutate()) for s in specimens
        ]


        image = images[0][0]
        s = np.array(softmaxes[np.argmax(losses)])
        s = np.exp(s) / np.sum(np.exp(s))
        exemplar_confidence = s[label]

    image = (image - image.min()) / (image.max() - image.min())
    image = np.array(image.detach().cpu())
    s = np.array(softmaxes[np.argmin(losses)])
    s = np.exp(s) / np.sum(np.exp(s))
    exemplar_confidence = s[label]

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


