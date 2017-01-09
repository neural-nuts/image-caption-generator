from ImgCap.generate_data import generate_captions
from random import shuffle
import tensorflow as tf
import numpy as np

class Caption_Generator():
    def __init__(self, data, dim_imgft=1536, embedding_size=256, num_hidden=256,
                 batch_size=100, num_timesteps=22,
                 word_threshold = 2, max_len = 20, resume=0):
        self.dim_imgft = np.int(dim_imgft)
        self.embedding_size = np.int(embedding_size)
        self.num_hidden = np.int(num_hidden)
        self.batch_size = np.int(batch_size)
        self.num_timesteps = np.int(num_timesteps)
        self.max_len=max_len
        self.word_threshold=word_threshold
        self.learning_rate=0.001
        self.wtoidx, self.idxtow, self.features, self.captions = data
        if resume is 0:
            self.idx = np.random.permutation(self.features.shape[0])
            self.captions = self.captions[self.idx]
            self.features = self.features[self.idx]
            self.batch_iter=self.get_next_batch()
        else:
            #load()
            pass
        self.max_words = np.int(len(self.wtoidx))
        self.model()


    def assign_weights(self, dim1, dim2 = None, name = None):
        return tf.Variable(tf.random_uniform([dim1, dim2], -0.1, 0.1),
                           name = name)
    def assign_biases(self, dim, name):
        return tf.Variable(tf.zeros([dim]), name = name)

    def model(self):
        self.word_embedding = {"weights" : self.assign_weights(self.max_words,
                                self.embedding_size, 'weight_emb'),
                               "biases": self.assign_biases(self.embedding_size,
                               "bias_emb")}

        self.image_embedding = {"weights" : self.assign_weights(self.dim_imgft,
                                self.num_hidden, 'weight_img_emb'),
                                "biases" : self.assign_biases(self.num_hidden,
                                'bias_img_emb')}

        self.target_word = {"weights" : self.assign_weights(self.num_hidden,
                            self.max_words, 'weight_target'),
                            "biases" : self.assign_biases(self.max_words,
                            'bias_target')}

        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden,
                                                      state_is_tuple=False)

        self.inp_dict = {"features":tf.placeholder(tf.float32,
                         [self.batch_size, self.dim_imgft]),
                         "captions":tf.placeholder(tf.int32,
                         [self.batch_size, self.num_timesteps]),
                         "mask":tf.placeholder(tf.float32,
                         [self.batch_size, self.num_timesteps])}

    def Words_to_IDs(self, wtoidx, caption_batch):
        for i,caption in enumerate(caption_batch):
            cap=[]
            for word in caption.split():
                try:
                    cap.append(wtoidx[word])
                except KeyError:
                    cap.append(1)
            caption_batch[i]=np.array(cap)
        return np.vstack(caption_batch)

    def IDs_to_Words(self, idxtow, ID_batch):
        arr=[]
        for sent in ID_batch:
            buf=''
            for word in sent:
                buf+=idxtow[word]+' '
            arr.append(buf.strip())
        return arr

    def generate_mask(self, ID_batch, wtoidx):
        nonpadded = map(lambda x: len(ID_batch[0])-x.count(wtoidx["<PAD>"]),ID_batch.tolist())
        ID_batch = np.zeros((ID_batch.shape[0], self.max_len+2))
        for ind, row in enumerate(ID_batch):
            row[:nonpadded[ind]] = 1
        return ID_batch

    def build_train_graph(self):
        initial_state = tf.zeros([self.batch_size, self.lstm_cell.state_size])
        image_emb = tf.matmul(self.inp_dict["features"], self.image_embedding['weights'])+ self.image_embedding['biases']

        with tf.variable_scope("LSTM"):
            output, state = self.lstm_cell(image_emb, initial_state)

        loss = 0.0
        with tf.variable_scope("LSTM"):
            for i in range(1, self.num_timesteps): # maxlen + 1
                batch_embed = tf.nn.embedding_lookup(self.word_embedding['weights'], self.inp_dict['captions'][:,i-1]) + self.word_embedding['biases']
                tf.get_variable_scope().reuse_variables()
                output, state = self.lstm_cell(batch_embed, state)
                words = tf.reshape(self.inp_dict['captions'][:, i], shape = [self.batch_size,1])
                onehot_encoded = tf.one_hot(indices = words, depth = len(self.wtoidx), on_value = 1, off_value = 0, axis = -1)
                onehot_encoded = tf.reshape(onehot_encoded, shape = [self.batch_size, self.max_words])
                target_logit = tf.matmul(output, self.target_word['weights']) + self.target_word['biases']
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(target_logit, onehot_encoded)
                cross_entropy = cross_entropy * self.inp_dict["mask"][:,i]
                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss

        loss = loss / tf.reduce_sum(self.inp_dict["mask"][:,1:])
        return loss, self.inp_dict
    

    def build_decode_graph(self):
        ##TODO
        image_features = tf.placeholder(tf.float32, [1, self.dim_imgft])

    def get_next_batch(self):
        batch_size=self.batch_size
        for batch_idx in range(0, len(self.features), batch_size):
            images_batch = self.features[batch_idx:batch_idx+batch_size]
            caption_batch = self.captions[batch_idx:batch_idx+batch_size]
            yield images_batch, caption_batch

    def train(self, loss, inp_dict):
        n_epochs=10
        self.loss=loss
        self.inp_dict=inp_dict
        saver = tf.train.Saver(max_to_keep=50)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        with tf.Session() as sess:
            for var in tf.trainable_variables():
                print var.op.name
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(n_epochs):
                batch_features, batch_captions = self.batch_iter.next()
                batch_Ids = self.Words_to_IDs(self.wtoidx, batch_captions)
                batch_mask = self.generate_mask(batch_Ids, self.wtoidx)
                _, current_loss = sess.run([optimizer, self.loss], feed_dict={self.inp_dict['features'] : batch_features, self.inp_dict['captions'] : batch_Ids, self.inp_dict['mask'] : batch_mask})
                print "Current Loss: ", current_loss
                print "Epoch ", epoch
                saver.save(sess, "./model/", global_step=epoch)
