# Image Caption Generator
### Work in Progress
Updates(Mar 12, 2017):
1. Added Dropout Layer for LSTM, Xavier Glorot Initializer for Weights
2. Significant Optimizations for Caption Generation i.e Decode Routine, computation time reduce from 3 seconds to 0.2 seconds
3. Functionality to Freeze Graphs and Merge them.
4. Direct Serving(Dual Graph and Single Graph) Routines in /util/
5. Explored and chose the fastest and most efficient Image Preprocessing Method.
5. Ported code to TensorFlow r1.0

Updates(Feb 27, 2017):
1. Added BLEU evaluation metric and batch processing of images to produce batches of captions.

Updates(Feb 25, 2017):
1. Added optimizations and one-time pre-processing of Flickr30K data
2. Changed to a faster Image Preprocessing method using OpenCV

Upcoming:
1. FIFO-queues in training
2. Attention-Model
3. Trained Models for Distribution.

A Neural Network based generative model for captioning images.

##Pre-Requisites:
1. Tensorflow r1.0
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
