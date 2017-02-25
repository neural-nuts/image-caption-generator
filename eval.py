from caption import *
import numpy as np

validation_data = np.load("Dataset/Validation_Data.npy")
features , captions = validation_data[:,0], validation_data[:,1]
features = np.array([feat.astype(float) for feat in features])
model = Caption_Generator(mode = 'test', batch_decode=True)
generated_captions=model.batch_decode(features)
with open("Dataset/Val_Generated_Captions.txt", 'w') as f:
    for cap in generated_captions:
        f.write(cap+"\n")
