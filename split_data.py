import os
import numpy as np
import pandas as pd

def split_data():
    df = pd.read_pickle('./Padded_Caps')
    #del df['Unnamed: 0']
    feat_7 = np.load('./features_7.npy')
    trainlist = open('Flickr8k_text/Flickr_8k.trainImages.txt').readlines()
    vallist = open('Flickr8k_text/Flickr_8k.devImages.txt').readlines()
    testlist = open('Flickr8k_text/Flickr_8k.testImages.txt').readlines()
    trainlist = [i.split('\n')[0] for i in trainlist]
    vallist = [i.split('\n')[0] for i in vallist]
    testlist = [i.split('\n')[0] for i in testlist]
    dftr = pd.DataFrame()
    dfv = pd.DataFrame()
    dfte = pd.DataFrame()

    trainfeat = []
    valfeat = []
    testfeat = []

    for i, row in df.iterrows():
        if row['image'] in trainlist:
            dftr = dftr.append(row, ignore_index=True)
            trainfeat.append(feat_7[i])
        if row['image'] in vallist:
            dfv = dfv.append(row, ignore_index=True)
            valfeat.append(feat_7[i])
        if row['image'] in testlist:
            dfte = dfte.append(row, ignore_index=True)
            testfeat.append(feat_7[i])
    trainfeat = np.array(trainfeat)
    valfeat = np.array(valfeat)
    testfeat = np.array(testfeat)

    np.save('train_features_7', trainfeat)
    np.save('val_features_7', valfeat)
    np.save('test_features_7', testfeat)
    dftr.to_pickle('train_cap')
    dfv.to_pickle('val_cap')
    dfte.to_pickle('test_cap')

split_data()
