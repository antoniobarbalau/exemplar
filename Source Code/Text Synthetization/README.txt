A Generic and Model-Agnostic ExemplarSynthetization Framework for Explainable AI

Text Synthetization



1. Download and prepare the IMDB dataset
```
python download_and_prepare_dataset.py
```

This script downloads the IMDB dataset and processes it into two .csv files for training and testing. These two files are used for training the classifier network. Afterwords, for each split, reviews are sentence-tokenized and preprocessed, to be used for training the LSTM-VAE network.

2. Train the classifier
```
python train_classifier.py
```

This script trains the classifier for 5 epochs and saves the model and tokenizer to disk.

3. Train the sentence generator
```
python train_generator.py
```
This script trains the classifier for 120 epochs and saves the models to disk. The training code is based on https://github.com/rohithreddy024/VAE-Text-Generation

4. Synthetize samples

```
python synthetize.py
```
This script runs the optimization process for the specified class. In the script, you can change the optimized class, and softmax temperature for text generation.
