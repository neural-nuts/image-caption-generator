from caption import *
from ImgCap.generate_data import generate_captions
word_threshold = 2
max_len = 20
vocab, wtoidx, training_data = generate_captions(
    word_threshold, max_len, 'Dataset/results_20130124.token', 'Dataset/features.npy')

features , captions = training_data[:,0], training_data[:,1]
features = np.array([feat.astype(float) for feat in features])

data = (vocab.tolist(), wtoidx.tolist(), features, captions)
model = Caption_Generator(data=data, resume=1)
loss, inp_dict = model.build_train_graph()
model.train(loss, inp_dict)
