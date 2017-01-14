# Image Caption Generator
## Work in Progress
A Neural Network based generative model for captioning images.

Download Pre-Trained InceptionV4 Tensorflow graph from DeepDetect available [here](https://deepdetect.com/models/tf/inception_v4.pb)

Put inception_v4.pb in CNNs folder

Put features.npy, paddedcaptions.csv, results_20130124.token in Dataset

Run train.py with parameter resume = 0, to start fresh training procedure. resume = 1 restores training process from latest saved model.

Run test.py with path=IMAGE_PATH to generate captions.
