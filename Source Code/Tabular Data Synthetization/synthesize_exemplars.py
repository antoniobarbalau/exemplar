from dataset import Adult
from dataset import Adult
from sklearn.ensemble import RandomForestClassifier
from vae import VAE
import numpy as np
import pickle
import torch

d = Adult()

device = torch.device('cpu')
vae = VAE().to(device)
vae.load_state_dict(torch.load('./adult_vae'))
vae.train(False)

cl = pickle.load(open('./rf.pkl', 'rb'))

for class_ in range(2):
    f = open(f'class_{class_}_exemplars.txt', 'w')
    for _ in range(100):
        encodings = np.random.uniform(-5., 5., size = (30, 4))
        c = 0.
        while c < .95:
            numerical, linear, categorical = vae.generate(
                torch.Tensor(encodings).to(device)
            )
            numerical = np.array(numerical.detach().cpu())
            linear = np.array(linear.detach().cpu())
            categorical = [
                np.array(elem.detach().cpu())
                for elem in categorical
            ]
            json_samples = d.postprocess(numerical, linear, categorical)
            samples = d.json_to_svm(json_samples)

            probabilities = cl.predict_proba(samples)
            losses = [np.sum(np.square(p - np.eye(2)[class_])) for p in probabilities]
            c = np.max([p[class_] for p in probabilities])
            indexes = np.argsort(losses)
            json_samples = [json_samples[i] for i in indexes]
            encodings = encodings[indexes[:10]]
            encodings = np.concatenate([
                encodings,
                encodings + np.random.normal(0., 1., size = (10, 4)),
                encodings + np.random.normal(0., 1., size = (10, 4)),
            ], axis = 0)
        f.write(json_samples[0].__repr__())
        f.write('\n\n')
    f.close()


