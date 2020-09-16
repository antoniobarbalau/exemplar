A Generic and Model-Agnostic ExemplarSynthetization Framework for Explainable AI

Image Synthetization



1. Download the FER2013 dataset CSV

Download and extract fer2013.csv from the fer2013.tar.gz archive into the dataset folder

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Afterwards run
```
python extract_fer.py
```
from the current folder

2. Train the classifier
```
python train_classifier.py
```


3. Train the variational autoencoder
```
python train_vae.py
```

4. Test convergence

```
python convergence_test_es.py
python convergence_test_es_momentum.py
python convergence_test_gd.py
```

5. Synthesize exemplars

```
python synthesize_exemplars_es_momentum.py 5
python synthesize_exemplars_gd.py 5
```

The first argument specifies the class for which to synthesize (default = 5).
Class indexes are: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.
