# [Deprecated] Image Caption Generator

**Notice: This project uses an older version of TensorFlow, and is no longer supported. Please consider using other latest alternatives.**

A Neural Network based generative model for captioning images.

## Checkout the android app made using this image-captioning-model: [Cam2Caption](https://github.com/neural-nuts/Cam2Caption) and [the associated paper](http://ieeexplore.ieee.org/document/8272660/).

### Work in Progress

###### Updates(Jan 14, 2018):
1. Some Code Refactoring.
2. Added MSCOCO dataset support.

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

###### To-Do(Open for Contribution):
1. FIFO-queues in training
2. Attention-Model
3. Trained Models for Distribution.

## Pre-Requisites:
1. Tensorflow r1.0
2. NLTK
3. pandas
4. Download Flickr30K OR MSCOCO images and captions.
5. Download Pre-Trained InceptionV4 Tensorflow graph from DeepDetect available [here](https://deepdetect.com/models/tf/inception_v4.pb)

## Procedure to Train and Generate Captions:
1. Clone the Repository to preserve Directory Structure
2. For flickr30k put results_20130124.token and Flickr30K images in flickr30k-images folder OR For MSCOCO put captions_val2014.json and MSCOCO images in COCO-images folder .
3. Put inception_v4.pb in ConvNets folder
4. Generate features(features.npy) corresponding to the images in the dataset folder by running-
    - For Flickr30K: `python convfeatures.py --data_path Dataset/flickr30k-images --inception_path ConvNets/inception_v4.pb`
    - For MSCOCO: `python convfeatures.py --data_path Dataset/COCO-images --inception_path ConvNets/inception_v4.pb`
3. To Train the model run-
    - For Flickr30K: `python main.py --mode train --caption_path ./Dataset/results_20130124.token --feature_path ./Dataset/features.npy --resume`
    - For MSCOCO: `python main.py --mode train --caption_path ./Dataset/captions_val2014.json --feature_path ./Dataset/features.npy --data_is_coco --resume`
4. To Generate Captions for an Image run
    - `python main.py --mode test --image_path VALID_PATH`
5. For usage as a python library see [Demo.ipynb](https://github.com/neural-nuts/image-caption-generator/blob/master/Demo.ipynb)

(see `python main.py -h` for more)

## Miscellaneous Notes:

### Freezing the encoder and decoder Graphs
1. It's necessary to save both encoder and decoder graphs while running test. This is a one-time necessary run before freezing the encoder/decoder.
    - `python main.py --mode test --image_path ANY_TEST_IMAGE.jpg/png --saveencoder --savedecoder`
2. In the project root directory use - `python utils/save_graph.py --mode encoder --model_folder model/Encoder/` additionally you may want to use `--read_file` if you want to freeze the encoder for directly generating caption for an image file(path). Similarly, for decoder use - `python utils/save_graph.py --mode decoder --model_folder model/Decoder/`, read_file argument is not necessary for the decoder.
3. To use frozen encoder and decoder models as dual blackbox [Serve-DualProtoBuf.ipynb](https://github.com/neural-nuts/image-caption-generator/blob/master/utils/Serve-DualProtoBuf.ipynb). Note: You must freeze encoder graph with --read_file to run this notebook

(see `python utils/save_graph.py -h` for more)

### Merging the encoder and decoder graphs for serving the model as a blackbox:
1. It's necessary to freeze the encoder and decoder as mentioned above.
2. In the project root directory run-
    - `python utils/merge_graphs.py --encpb ./model/Trained_Graphs/encoder_frozen_model.pb --decpb ./model/Trained_Graphs/decoder_frozen_model.pb` additionally you may want to use `--read_file` if you want to freeze the encoder for directly generating caption for an image file(path).
3. To use merged encoder and decoder models as single frozen blackbox: [Serve-SingleProtoBuf.ipynb](https://github.com/neural-nuts/image-caption-generator/blob/master/utils/Serve-SingleProtoBuf.ipynb). Note: You must freeze and merge encoder graph with --read_file to run this notebook

(see `python utils/merge_graphs.py -h` for more)

### Training Steps vs Loss Graph in Tensorboard:
1. `tensorboard --logdir model/log_dir`
2. Navigate to `localhost:6006`

## Citation:

If you use our model or code in your research, please cite the paper:

```
@article{Mathur2017,
  title={Camera2Caption: A Real-time Image Caption Generator},
  author={Pranay Mathur and Aman Gill and Aayush Yadav and Anurag Mishra and Nand Kumar Bansode},
  journal={IEEE Conference Publication},
  year={2017}
}
```

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
