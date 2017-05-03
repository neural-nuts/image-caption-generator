import nltk
import pandas as pd
import numpy as np


path_to_reference = 'Dataset/Validation_Captions.txt' # df -> image_id:str     caption:str     len(5000)
path_to_model = 'model/Decoder/Generated_Captions.txt'

with open(path_to_model) as f:
  model_data = f.readlines()
model_filenames=[caps.split('\t')[0] for caps in model_data]
model_captions = [caps.replace('\n', '').split('\t')[1] for caps in model_data]

with open(path_to_reference, 'r') as f:
    ref_data = f.readlines()
reference_filenames = [caps.split('\t')[0].split('#')[0] for caps in ref_data]
reference_captions = [caps.replace('\n', '').split('\t')[1] for caps in ref_data]

df = pd.DataFrame()
df['image'] = reference_filenames
df['caption'] = reference_captions
df.caption = df.caption.str.decode('utf').str.split()
df = pd.DataFrame(data = {'image':list(df.image.unique()),'caption':list(df.groupby('image')['caption'].apply(list))})[:len(model_captions)]

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []
index1=None
index2=None

for i, row in df.iterrows():
  model = model_captions[i].split()
  reference = row.caption
  try:
      score1 = nltk.translate.bleu_score.sentence_bleu(reference, model, weights=[1.0])
      score2 = nltk.translate.bleu_score.sentence_bleu(reference, model, weights=[0.5,0.5])
      score3 = nltk.translate.bleu_score.sentence_bleu(reference, model, weights=[1.0/3,1.0/3,1-2*(1.0/3)])
      score4 = nltk.translate.bleu_score.sentence_bleu(reference, model)
      bleu1_scores.append(score1)
      bleu2_scores.append(score2)
      bleu3_scores.append(score3)
      bleu4_scores.append(score4)
      if i%10000 == 0 and i!=0:
          print (float(i)/df.shape[0])*100,"%"," done"
  except:
      index1=df.index[i]
      index2=i
      print "Invalid Caption Generated for: ", model_filenames[i]

print "\nMean Sentence-Level BLEU-1 score: ", np.mean(bleu1_scores)
print "Mean Sentence-Level BLEU-2 score: ", np.mean(bleu2_scores)
print "Mean Sentence-Level BLEU-3 score: ", np.mean(bleu3_scores)
print "Mean Sentence-Level BLEU-4 score: ", np.mean(bleu4_scores)


if index1 and index2:
    df=df.drop([index1])
    df=df.reset_index(drop=True)
    del model_captions[index2]

references=df.caption
model_captions = [caption.split() for caption in model_captions]

score1 = nltk.translate.bleu_score.corpus_bleu(references,model_captions, weights=[1.0])
print "\n\nCorpus-Level BLEU-1 score: ", score1
score2 = nltk.translate.bleu_score.corpus_bleu(references,model_captions, weights=[0.5,0.5])
print "Corpus-Level BLEU-2 score: ", score2
score3 = nltk.translate.bleu_score.corpus_bleu(references,model_captions, weights=[1.0/3,1.0/3,1-2*(1.0/3)])
print "Corpus-Level BLEU-3 score: ", score3
score4 = nltk.translate.bleu_score.corpus_bleu(references,model_captions)
print "Corpus-Level BLEU-4 score: ", score4
