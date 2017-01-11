import numpy as np
import pandas as pd
from collections import Counter

max_len = 20
word_threshold = 2

def preprocess_captions(filenames,captions):
    global max_len
    print "Preprocessing Captions"
    df = pd.DataFrame()
    df['FileNames'] = filenames
    df['caption'] = captions
    df['caption'] = df['caption'].replace(r'(\,|\.|\#|\*|\%|\$|\@|\!|\:|\;|\~|\`|\?|\'|\"|\(|\))','',regex = True).replace(r'\&','and',regex = True).replace('-',' ',regex = True)
    df['caption'] = df['caption'].str.split().apply(lambda x: x[:max_len]).apply(" ".join).str.lower()
    df = df[:158900]
    return df

def generate_vocab(df):
    global max_len, word_threshold
    print "Generating Vocabulary"
    counter = Counter()
    for i, row in df.iterrows():
        counter.update(row["caption"].lower().split())
    vocab = dict([w for w in counter.items() if w[1] >= word_threshold])
    wtoidx = {}
    idxtow = {}
    wtoidx["<S>"] = 0 #Start
    wtoidx["<PAD>"] = 1 #PAD
    wtoidx["</S>"] = 2 #End
    i = 3
    print "Generating Word to Index and Index to Word"
    for word in vocab.keys():
        wtoidx[word] = i
        i = i+1
    idxtow = dict(zip(wtoidx.values(),wtoidx.keys()))
    print "Size of Vocabulary", len(vocab)
    return vocab, wtoidx, idxtow

def pad_captions(df):
    global max_len
    print "Padding Caption <PAD> to Max Length", max_len, "+ 2 for <S> and </S>"
    dfPadded = df.copy()
    dfPadded['caption'] = "<S> "+dfPadded['caption']+" </S>"
    max_len = max_len+2
    for i, row in dfPadded.iterrows():
        cap = row['caption']
        cap_len = len(cap.split())
        if(cap_len < max_len):
            pad_len = max_len-cap_len
            pad_buf = "<PAD> " * pad_len
            pad_buf = pad_buf.strip()
            dfPadded.set_value(i, 'caption', cap+" "+pad_buf)
    print "Saving Padded Captions to paddedcaptions.csv"
    dfPadded.to_csv('Dataset/paddedcaptions.csv',index = False)
    return dfPadded

def load_features(feature_path):
    features = np.load(feature_path)
    features = np.repeat(features, 5, axis = 0)
    print "Features Loaded", feature_path
    return features


def generate_captions(wt = 2, ml = 20, cap_path = 'Dataset/results_20130124.token', feat_path =  'Dataset/features.npy', mode = 'train'):
    global max_len, word_threshold
    max_len = ml
    word_threshold = wt
    print "Loading Caption Data", cap_path
    with open(cap_path,'r') as f:
        data = f.readlines()
    filenames = [caps.split('\t')[0].split('#')[0] for caps in data]
    captions = [caps.replace('\n','').split('\t')[1] for caps in data]
    df = preprocess_captions(filenames,captions)
    vocab, wtoidx, idxtow = generate_vocab(df)
    if mode is 'test':
        return wtoidx, idxtow
    df = pad_captions(df)
    features = load_features(feat_path)
    captions = np.array(df.caption)
    print "Preprocessing Complete"
    return vocab, wtoidx, idxtow, features, captions
