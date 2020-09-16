A Generic and Model-Agnostic ExemplarSynthetization Framework for Explainable AI


We provide a minimal source code able to run our experiments. Each data modality has usage guidelines inside their respective folder.

Additionally, we provide image synsthetization videos in which we feature the evolution of population during the Evolutionary Strategy optimization. Synthesized tabular examples are also provided within the Tabular Exemplars folder.

The requirements for the tabular and image synthetization code are the following:

pytorch, torchvision, pillow<7, opencv-python, matplotlib, sklearn

Except for Pillow which requires a version lower than 7, we used the latest versions of all other libraries. We also specify that our models require a GPU in order to run.

If using Anaconda the following commands will generate a sufficient environment for running the code:


conda create -n env python=3.7.6
conda install pytorch
conda install torchvision
pip install "pillow<7"
pip install opencv-python
pip install matplotlib
pip install sklearn


For text generation, requirements are presented in text_requirements.txt.
