A Generic and Model-Agnostic ExemplarSynthetization Framework for Explainable AI

Tabular Data Synthetization



1. Download the adult.csv dataset from https://www.kaggle.com/wenruliu/adult-income-dataset inside the dataset folder


2. Train the classifier
```
python train_classifier.py
```

3. Train the variational autoencoder
```
python train_vae.py
```

5. Synthesize exemplars

```
python synthesize_exemplars.py
```

Synthesized exemplars will be provided in 2 separate files: class_0_exemplars.txt and class_1_exemplars.txt. Where class 0 represents the low income class and class 1 represents the high income class.
