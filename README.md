# Image Caption Generator
### Work in Progress
A Neural Network based generative model for captioning images.

##Pre-Requisites:
1. Tensorflow
2. NLTK
3. pandas
4. Download Flickr30K Dataset: Images and results_20130124.token
5. Download Pre-Trained InceptionV4 Tensorflow graph from DeepDetect available [here](https://deepdetect.com/models/tf/inception_v4.pb)

##Procedure to Run:
1. Clone the Repository to preserve Directory Structure
1. Put results_20130124.token and Flickr30K images in Dataset Folder.
2. Put inception_v4.pb in ConvNets folder
3. To Train the model run main.py with --mode train and --resume 0 or 1. 
4. To Generate Captions for an Image run main.py with --mode test and --image_path VALID_PATH.

(see python main.py -h for more)

##Reference:
Show and Tell: A Neural Image Caption Generator

-Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan

##License:
Protected Under BSD-3 Clause License.
