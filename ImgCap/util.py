import numpy as np

def Words_to_IDs(wtoidx, caption_batch):
    return np.array(map(lambda cap: [wtoidx[word] for word in cap.split(' ') if word in wtoidx.keys()], caption_batch))

def IDs_to_Words(idxtow, ID_batch):
    arr=[]
    for sent in ID_batch:
        buf=''
        for word in sent:
            buf+=idxtow[word]+' '
        arr.append(buf.strip())
    return arr

def generate_mask(ID_batch, wtoidx):
    nonpadded = map(lambda x: len(ID_batch[0])-x.count(wtoidx["<PAD>"]),ID_batch.tolist())
    ID_batch = np.zeros((ID_batch.shape[0], ID_batch.shape[1]))
    for ind, row in enumerate(ID_batch):
        row[:nonpadded[ind]] = 1
    return np.array(ID_batch)
