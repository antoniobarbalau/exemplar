from classifier import VGG
from torch import nn
from vae import VAE
import cv2
import numpy as np
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
        alpha = 0.
        velocity = np.random.normal(scale = .5, size = (256,))
        if self.momentum is not None:
            velocity = alpha * self.momentum + (1 - alpha) * velocity
        return (
            self.position + velocity,
            velocity
        )

def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    return np.sum(np.power(
        softmax -
        np.eye(7)[label],
        2
    ))

iterations = 0
n_tests = 1000
for test_n in range(n_tests):
    exemplar_confidence = 0.
    specimens = [
        Specimen() for _ in range(50)
    ]
    label = np.random.randint(7)
    while exemplar_confidence < .95:
        iterations += 1
        encodings = np.array([
            s.position for s in specimens
        ], dtype = np.float32)
        images = generator.generate(
            torch.Tensor(encodings).cuda()
        )
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
    print(f'Synthesizing exemplar: {test_n} / {n_tests}   ', end = '\r')

print(f'Mean number of iterations: {iterations / n_tests}')
print(f'Mean number of forward passes required: {(iterations / n_tests) * 20 + 50}')


