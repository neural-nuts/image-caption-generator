import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import pickle
import os

max_len = 20
word_threshold = 2
counter = None


def preprocess_captions(filenames, captions):
    global max_len
    print "Preprocessing Captions"
    df = pd.DataFrame()
    df['FileNames'] = filenames
    df['caption'] = captions
    df.caption = df.caption.str.decode('utf')
    df['caption'] = df['caption'].apply(word_tokenize).apply(
        lambda x: x[:max_len]).apply(" ".join).str.lower()
    #df = df[:158900] #uncomment if flickr
    return df


def generate_vocab(df):
    global max_len, word_threshold, counter
    print "Generating Vocabulary"

    vocab = dict([w for w in counter.items() if w[1] >= word_threshold])
    vocab["<UNK>"] = len(counter) - len(vocab)
    vocab["<PAD>"] = df.caption.str.count("<PAD>").sum()
    vocab["<S>"] = df.caption.str.count("<S>").sum()
    vocab["</S>"] = df.caption.str.count("</S>").sum()
    wtoidx = {}
    wtoidx["<S>"] = 1
    wtoidx["</S>"] = 2
    wtoidx["<PAD>"] = 0
    wtoidx["<UNK>"] = 3
    print "Generating Word to Index and Index to Word"
    i = 4
    for word in vocab.keys():
        if word not in ["<S>", "</S>", "<PAD>", "<UNK>"]:
            wtoidx[word] = i
            i += 1
    print "Size of Vocabulary", len(vocab)
    return vocab, wtoidx


def pad_captions(df):
    global max_len
    print "Padding Caption <PAD> to Max Length", max_len, "+ 2 for <S> and </S>"
    dfPadded = df.copy()
    dfPadded['caption'] = "<S> " + dfPadded['caption'] + " </S>"
    max_len = max_len + 2
    for i, row in dfPadded.iterrows():
        cap = row['caption']
        cap_len = len(cap.split())
        if(cap_len < max_len):
            pad_len = max_len - cap_len
            pad_buf = "<PAD> " * pad_len
            pad_buf = pad_buf.strip()
            dfPadded.set_value(i, 'caption', cap + " " + pad_buf)
    return dfPadded


def load_features(feature_path):
    features = np.load(feature_path)
    features = np.repeat(features, 5, axis=0)
    print "Features Loaded", feature_path
    return features


def split_dataset(df, features, ratio=0.8):
    split_idx = int(df.shape[0] * ratio)
    print "Data Statistics:"
    print "# Records Total Data: ", df.shape[0]
    print "# Records Training Data: ", split_idx
    print "# Records Training Data: ", df.shape[0] - split_idx
    print "Ration of Training: Validation = ", ratio * 100, ":", 100 - (ratio * 100)
    val_features = features[split_idx:]
    val_captions = np.array(df.caption)[split_idx:]
    np.save("Dataset/Validation_Data", zip(val_features, val_captions))
    return df[:split_idx], features[:split_idx]


def get_data(required_files):
    ret = []
    for fil in required_files:
        ret.append(np.load("Dataset/" + fil + ".npy"))
    return ret


def generate_captions(
        wt=2,
        ml=20,
        cap_path='Dataset/results_20130124.token',
        feat_path='Dataset/features.npy'):
    required_files = ["vocab", "wordmap", "Training_Data"]
    generate = False
    for fil in required_files:
        if not os.path.isfile('Dataset/' + fil + ".npy"):
            generate = True
            print "Required Files not present. Regenerating Data."
            break
    if not generate:
        print "Dataset Present; Skipping Generation."
        return get_data(required_files)
    global max_len, word_threshold, counter
    max_len = ml
    word_threshold = wt
    print "Loading Caption Data", cap_path
    with open(cap_path, 'r') as f:
        data = f.readlines()
    filenames = [caps.split('\t')[0].split('#')[0] for caps in data]
    captions = [caps.replace('\n', '').split('\t')[1] for caps in data]
    df = preprocess_captions(filenames, captions)
    features = load_features(feat_path)
    idx = np.random.permutation(features.shape[0])
    df = df.iloc[idx]
    features = features[idx]
    # df, features = split_dataset(df, features) #use flickr8k for
    # validationSSS
    counter = Counter()
    for i, row in df.iterrows():
        counter.update(row["caption"].lower().split())
    df = pad_captions(df)
    vocab, wtoidx = generate_vocab(df)
    captions = np.array(df.caption)
    np.save("Dataset/Training_Data", zip(features, captions))
    np.save("Dataset/wordmap", wtoidx)
    np.save("Dataset/vocab", vocab)

    print "Preprocessing Complete"
    return get_data(required_files)
