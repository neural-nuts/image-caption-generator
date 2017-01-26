# Image Caption Generator
### Work in Progress
A Neural Network based generative model for captioning images.

##Pre-Requisites:
1. Tensorflow
2. NLTK
3. pandas
4. skimage
5. Download Flickr30K Dataset: Images and results_20130124.token
6. Download Pre-Trained InceptionV4 Tensorflow graph from DeepDetect available [here](https://deepdetect.com/models/tf/inception_v4.pb)

##Procedure to Run:
1. Clone the Repository to preserve Directory Structure
1. Put results_20130124.token and Flickr30K images in Dataset Folder.
2. Put inception_v4.pb in CNNs folder
3. Run train.py with parameter resume = 0, to start fresh training procedure. resume = 1 restores training process from latest saved model.
4. Run test.py with path=IMAGE_PATH/URL to generate captions OR follow test.ipynb

##Reference:
Show and Tell: A Neural Image Caption Generator

-Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
