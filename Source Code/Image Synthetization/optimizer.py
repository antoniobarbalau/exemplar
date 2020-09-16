import numpy as np
import time
import torch

def loss(logits, label):
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    return np.sum(np.power(softmax - np.eye(7)[label], 2))

def plot(images, wait = 10):
    images = images / 2. + .5
    images = np.reshape(images, (4, 4, 48, 48))
    images = np.hstack(images)
    images = np.hstack(images)
    cv2.imshow('frame', images)
    cv2.waitKey(wait)


class ESMomentum(object):
    def __init__(
        self,
        classifier,
        generator,
        device,
        mu = .3,
        u = 5.,
        t = 50,
        m = 2,
        k = 10,
        scale = .5,
        embedding_size = 256,
        confidence = .95,
        visualize = False,
        frame_interval = 10
    ):
        self.classifier = classifier
        self.generator = generator
        self.device = device
        self.mu = mu
        self.u = u
        self.t = t
        self.scale = scale
        self.m = m
        self.k = k
        self.embedding_size = embedding_size
        self.confidence = confidence
        self.visualize = visualize
        self.frame_interval = frame_interval


    def synthesize(self, label):
        start_time = time.time()

        encodings = np.random.uniform(
            -self.u, self.u,
            size = (self.t, self.embedding_size)
        )
        moments = np.array([np.zeros(self.embedding_size) for _ in range(self.t)])
        images = self.generator.generate(
            torch.Tensor(encodings).to(self.device)
        )

        with torch.no_grad():
            logits = np.array(self.classifier(images).detach().cpu())
        images = np.array(images.detach().cpu())
        losses = np.array([loss(l, label) for l in logits])
        indexes = np.argsort(losses)[:self.k]

        exemplar_logits = np.array(logits[np.argmax(losses)])
        exemplar_softmax = np.exp(exemplar_logits) / np.sum(np.exp(exemplar_logits))
        confidence = exemplar_softmax[label]

        encodings = encodings[indexes]
        moments = moments[indexes]
        images = images[indexes]
        losses = losses[indexes]
        logits = logits[indexes]

        n_iterations = 0
        while confidence < self.confidence:
            n_iterations += 1
            new_encodings = []
            new_moments = []
            for specimen_n in range(self.k):
                for _ in range(self.m):
                    velocity = np.random.normal(
                        scale = self.scale,
                        size = [self.embedding_size]
                    )
                    if np.any(moments[specimen_n] != 0):
                        velocity = (
                            self.mu * moments[specimen_n] +
                            (1. - self.mu) * velocity
                        )
                    new_encodings.append(encodings[specimen_n] + velocity)
                    new_moments.append(velocity)
            new_encodings = np.array(new_encodings)
            new_moments = np.array(new_moments)
            new_images = self.generator.generate(
                torch.Tensor(new_encodings).to(self.device)
            )
            with torch.no_grad():
                new_logits = np.array(self.classifier(new_images).detach().cpu())
            logits = np.concatenate([logits, np.array(new_logits)], axis = 0)
            losses = np.concatenate([
                losses,
                np.array([loss(l, label) for l in new_logits])
            ])
            encodings = np.concatenate([
                encodings, new_encodings
            ], axis = 0)
            moments = np.concatenate([
                moments, new_moments
            ])
            images = np.concatenate([
                images, np.array(new_images.detach().cpu())
            ], axis = 0)
            indexes = np.argsort(losses)
            images = images[indexes]

            if self.visualize:
                plot(images[:16])
            
            exemplar_logits = np.array(logits[np.argmax(losses)])
            exemplar_softmax = np.exp(exemplar_logits) / np.sum(np.exp(exemplar_logits))
            confidence = exemplar_softmax[label]

            indexes = indexes[:self.k]
            images = images[:self.k]
            encodings = encodings[indexes]
            moments = moments[indexes]
            losses = losses[indexes]
            logits = logits[indexes]
        return images[0], n_iterations, time.time() - start_time

