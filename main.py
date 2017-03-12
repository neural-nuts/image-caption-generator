from caption_generator import *
from utils.data_util import generate_captions
from configuration import Configuration
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    help="train|test|eval",
    choices=[
        "train",
        "test",
        "eval"],
    required=True)
parser.add_argument(
    "--resume",
    type=int,
    help="1=Yes|0=No",
    choices=[
        1,
        0],
    default=0)
parser.add_argument(
    "--saveencoder",
    type=int,
    help="1=Yes|0=No",
    choices=[
        1,
         0])
parser.add_argument(
    "--savedecoder",
    type=int,
    help="1=Yes|0=No",
    choices=[
        1,
         0])
parser.add_argument(
    "--image_path",
    type=str,
    help="Path to the Image for Generation of Captions")
parser.add_argument(
    "--validation_data",
    type=str,
    help="Path to the Validation Data for evaluation")
args = parser.parse_args()
config = Configuration(vars(args))

if config.mode == "train":
    caption_file = 'Dataset/results_20130124.token'
    feature_file = 'Dataset/features.npy'
    vocab, wtoidx, training_data = generate_captions(
        config.word_threshold, config.max_len, caption_file, feature_file)
    features, captions = training_data[:, 0], training_data[:, 1]
    features = np.array([feat.astype(float) for feat in features])
    data = (vocab.tolist(), wtoidx.tolist(), features, captions)
    model = Caption_Generator(config, data=data)
    loss, inp_dict = model.build_train_graph()
    model.train(loss, inp_dict)

elif config.mode == "test":
    if os.path.exists(args.image_path):
        model = Caption_Generator(config)
        model.decode(args.image_path)
    else:
        print "Please provide a valid image path.\n Usage:\n python main.py --mode test --image_path VALID_PATH"

elif config.mode == "eval":
    config.mode = "test"
    config.batch_decode = True
    if os.path.exists(args["validation_data"]):
        validation_data = np.load(args["validation_data"])
        features, captions = validation_data[:, 0], validation_data[:, 1]
        features = np.array([feat.astype(float) for feat in features])
        model = Caption_Generator(config)
        generated_captions = model.batch_decode(features)
        with open("Dataset/Val_Generated_Captions.txt", 'w') as f:
            for cap in generated_captions:
                f.write(cap + "\n")
        path_to_reference = 'Dataset/Val_Captions' # df -> image_id:str     caption:str     len(5000)
        path_to_model = 'Dataset/Val_Generated_Captions.txt'
        with open(path_to_model) as f:
            model_summaries = f.readlines()

        df = pd.read_pickle(path_to_reference)
        df = pd.DataFrame(data = {'image':list(df.images.unique()),'caption':list(df.groupby('image')['caption'].apply(list))})
        #df -> image_id:str     caption: list of str    len(1000)
        bleu_scores = []

        for i, row in df.iterrows():
            model = model_summaries[i]
            reference = row.caption
            bleu_scores.append(nltk.translate.bleu_score.sentence_bleu(reference, model, weights=[0.4,0.3,0.2]))

        print "Mean BLEU score: ", np.mean(bleu_score)
    else:
        print "Please provide a valid path to Validation Data.\n Usage:\n python main.py --mode eval --validation_data VALID_PATH"
