# Image Caption Generator

A Neural Network based generative model for captioning images.

Checkout the android app made using this image-captioning-model: [Cam2Caption](https://github.com/neural-nuts/Cam2Caption)
### Work in Progress
###### Updates(Mar 12, 2017):
1. Added Dropout Layer for LSTM, Xavier Glorot Initializer for Weights
2. Significant Optimizations for Caption Generation i.e Decode Routine, computation time reduce from 3 seconds to 0.2 seconds
3. Functionality to Freeze Graphs and Merge them.
4. Direct Serving(Dual Graph and Single Graph) Routines in /util/
5. Explored and chose the fastest and most efficient Image Preprocessing Method.
5. Ported code to TensorFlow r1.0

###### Updates(Feb 27, 2017):
1. Added BLEU evaluation metric and batch processing of images to produce batches of captions.

###### Updates(Feb 25, 2017):
1. Added optimizations and one-time pre-processing of Flickr30K data
2. Changed to a faster Image Preprocessing method using OpenCV

###### Upcoming:
1. FIFO-queues in training
2. Attention-Model
3. Trained Models for Distribution.

## Pre-Requisites:
1. Tensorflow r1.0
2. NLTK
3. pandas
4. Download Flickr30K Dataset: Images and results_20130124.token
5. Download Pre-Trained InceptionV4 Tensorflow graph from DeepDetect available [here](https://deepdetect.com/models/tf/inception_v4.pb)

## Procedure to Run:
1. Clone the Repository to preserve Directory Structure
2. Put results_20130124.token and Flickr30K images in dataset folder.
3. Put inception_v4.pb in ConvNets folder
4. Generate features(features.npy) corresponding to the images in the dataset folder by running:  python convfeatures.py .
3. To Train the model run main.py with --mode train and --resume 0 or 1. 
4. To Generate Captions for an Image run main.py with --mode test and --image_path VALID_PATH.

(see python main.py -h for more)

## Reference:
Show and Tell: A Neural Image Caption Generator

-Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan

## License:
Protected Under BSD-3 Clause License.

## Some Examples:

![Alt text](/Images/gen_3126981064.jpg)
![Alt text](/Images/gen_7148046575.jpg)
![Alt text](/Images/gen_suitselfie.png)
![Alt text](/Images/gen_6.png)
![Alt text](/Images/gen_7526599338.jpg)
![Alt text](/Images/gen_4013421575.jpg)
![Alt text](/Images/gen_football.png)
![Alt text](/Images/gen_plane.png)
![Alt text](/Images/gen_comp.png)
![Alt text](/Images/gen_womanbeach.png)
![Alt text](/Images/gen_102617084.jpg)
![Alt text](/Images/gen_2230458748.jpg)
![Alt text](/Images/gen_7125476937.jpg)
![Alt text](/Images/gen_4752984291.jpg)
![Alt text](/Images/gen_cat2.png)
![Alt text](/Images/gen_283252248.jpg)
![Alt text](/Images/gen_3920626767.jpg)
![Alt text](/Images/gen_manlaptop.png)
![Alt text](/Images/gen_2461372011.jpg)
