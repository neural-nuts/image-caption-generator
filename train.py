from caption import *
word_threshold=2
max_len=20
_, wtoidx, idxtow, features, captions = generate_captions(word_threshold, max_len, 'Dataset/results_20130124.token', 'Dataset/features.npy')
data=(wtoidx, idxtow, features, captions)
model = Caption_Generator(data = data, resume = 1)
loss, inp_dict= model.build_train_graph()
model.train(loss, inp_dict)
