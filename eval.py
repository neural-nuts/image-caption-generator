from caption import *
import numpy as np
import pandas as pd

#Flickr30k validation: 1000 images
path_to_reference = 'Dataset/Val_Captions' # df -> image_id:str     caption:str     len(5000)
path_to_model = 'Dataset/Val_Generated_Captions.txt'

validation_data = np.load("Dataset/Validation_Data.npy") # non-replicated features   len(1000)
#features , captions = validation_data[:,0], validation_data[:,1]
#features = np.array([feat.astype(float) for feat in features])
model = Caption_Generator(mode = 'test', batch_decode=True)
generated_captions=model.batch_decode(validation_data) # str    len(1000)
with open(path_to_model, 'w') as f:
    for cap in generated_captions:
        f.write(cap+"\n")

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
