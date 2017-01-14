from ImgCap.generate_data import generate_captions
from random import shuffle
from CNN import *
import tensorflow as tf
import numpy as np
import sys
import os


class Caption_Generator():

    def __init__(
            self,
            dim_imgft=1536,
            embedding_size=256,
            num_hidden=256,
            batch_size=100,
            num_timesteps=22,
            data=None,
            mode='train',
            word_threshold=2,
            max_len=20,
            resume=0):
        self.dim_imgft = np.int(dim_imgft)
        self.embedding_size = np.int(embedding_size)
        self.num_hidden = np.int(num_hidden)
        self.batch_size = np.int(batch_size)
        self.num_timesteps = np.int(num_timesteps)
        self.max_len = max_len
        self.word_threshold = word_threshold
        self.mode = mode
        self.learning_rate = 0.001
        if self.mode == 'train':
            self.wtoidx, self.idxtow, self.features, self.captions = data
            self.idx = np.random.permutation(self.features.shape[0])
            self.num_batch = int(self.features.shape[0]) / self.batch_size
            print "Converting Captions to IDs"
            self.captions = self.Words_to_IDs(self.wtoidx, self.captions)

        self.resume = resume

        self.current_epoch = 0
        self.current_step = 0
        if self.resume is 1 or mode == 'test':
            if os.path.isfile(
                    'model/save.npy') and os.path.isfile('model/idx.npy'):
                self.current_epoch, self.current_step = np.load("model/save.npy") #
                self.idx = np.load("model/idx.npy")
            else:
                print "No Checkpoints, Restarting Training.."
                self.resume = 0
        if self.mode == "train":
            np.save("model/idx", self.idx)
        self.nb_epochs = 10000
        if self.mode == 'train':
            self.captions = self.captions[self.idx]
            self.features = self.features[self.idx]
        if self.mode == 'test':
            self.wtoidx, self.idxtow = generate_captions(
                self.word_threshold, self.max_len, mode='test')
        self.max_words = np.int(len(self.wtoidx))

        self.model()

    def Words_to_IDs(self, wtoidx, caption_batch):
        for i, caption in enumerate(caption_batch):
            cap = []
            for word in caption.split():
                try:
                    cap.append(wtoidx[word])
                except KeyError:
                    cap.append(1)
            caption_batch[i] = np.array(cap)
        return np.vstack(caption_batch)

    def IDs_to_Words(self, idxtow, ID_batch):
        arr = []
        for sent in ID_batch:
            buf = ''
            for word in sent:
                buf += idxtow[word] + ' '
            arr.append(buf.strip())
        return arr

    def generate_mask(self, ID_batch, wtoidx):
        nonpadded = map(lambda x: len(
            ID_batch[0]) - x.count(wtoidx["<PAD>"]), ID_batch.tolist())
        ID_batch = np.zeros((ID_batch.shape[0], self.max_len + 2))
        for ind, row in enumerate(ID_batch):
            row[:nonpadded[ind]] = 1
        return ID_batch

    def get_next_batch(self):
        batch_size = self.batch_size
        for batch_idx in range(0, len(self.features), batch_size):
            images_batch = self.features[batch_idx:batch_idx + batch_size]
            caption_batch = self.captions[batch_idx:batch_idx + batch_size]
            # print caption_batch
            yield images_batch, caption_batch

    def assign_weights(self, dim1, dim2=None, name=None):
        return tf.Variable(tf.random_uniform([dim1, dim2], -0.1, 0.1),
                           name=name)

    def assign_biases(self, dim, name):
        return tf.Variable(tf.zeros([dim]), name=name)

    def model(self):
        self.word_embedding = {
            "weights": self.assign_weights(
                self.max_words,
                self.embedding_size,
                'weight_emb'),
            "biases": self.assign_biases(
                self.embedding_size,
                "bias_emb")}

        self.image_embedding = {
            "weights": self.assign_weights(
                self.dim_imgft,
                self.num_hidden,
                'weight_img_emb'),
            "biases": self.assign_biases(
                self.num_hidden,
                'bias_img_emb')}

        self.target_word = {
            "weights": self.assign_weights(
                self.num_hidden,
                self.max_words,
                'weight_target'),
            "biases": self.assign_biases(
                self.max_words,
                'bias_target')}

        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden,
                                                      state_is_tuple=False)

        self.inp_dict = {
            "features": tf.placeholder(
                tf.float32, [self.batch_size, self.dim_imgft]),
            "captions": tf.placeholder(
                tf.int32, [self.batch_size, self.num_timesteps]),
            "mask": tf.placeholder(
                tf.float32, [self.batch_size, self.num_timesteps])
        }

    def create_feed_dict(self, Ids, features, mask, mode="train"):
        feed_dict = {}
        feed_dict[self.inp_dict['captions']] = Ids
        feed_dict[self.inp_dict['features']] = features
        feed_dict[self.inp_dict['mask']] = mask
        return feed_dict

    def build_train_graph(self):
        initial_state = tf.zeros([self.batch_size, self.lstm_cell.state_size])
        image_emb = tf.matmul(self.inp_dict["features"], self.image_embedding[
                              'weights']) + self.image_embedding['biases']

        with tf.variable_scope("LSTM"):
            output, state = self.lstm_cell(image_emb, initial_state)
            loss = 0.0
            for i in range(1, self.num_timesteps):  # maxlen + 1
                batch_embed = tf.nn.embedding_lookup(
                    self.word_embedding['weights'], self.inp_dict['captions'][
                        :, i - 1]) + self.word_embedding['biases']
                tf.get_variable_scope().reuse_variables()
                output, state = self.lstm_cell(batch_embed, state)
                words = tf.reshape(self.inp_dict['captions'][
                                   :, i], shape=[self.batch_size, 1])
                onehot_encoded = tf.one_hot(indices=words, depth=len(
                    self.wtoidx), on_value=1, off_value=0, axis=-1)
                onehot_encoded = tf.reshape(onehot_encoded, shape=[
                                            self.batch_size, self.max_words])
                target_logit = tf.matmul(
                    output, self.target_word['weights']) + self.target_word['biases']
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    target_logit, onehot_encoded)
                cross_entropy = cross_entropy * self.inp_dict["mask"][:, i]
                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss

        loss = loss / tf.reduce_sum(self.inp_dict["mask"][:, 1:])
        return loss, self.inp_dict

    def build_decode_graph(self):
        image_features = tf.placeholder(tf.float32, [1, self.dim_imgft])
        image_emb = tf.matmul(image_features, self.image_embedding[
                              'weights']) + self.image_embedding['biases']
        initial_state = tf.zeros([1, self.lstm_cell.state_size])
        IDs = []
        with tf.variable_scope("LSTM"):
            output, state = self.lstm_cell(image_emb, initial_state)
            last_word = tf.nn.embedding_lookup(
                self.word_embedding['weights'], [
                    self.wtoidx["<S>"]]) + self.word_embedding['biases']
            for i in range(self.num_timesteps):
                tf.get_variable_scope().reuse_variables()
                output, state = self.lstm_cell(last_word, state)
                logits = tf.matmul(output, self.target_word[
                                   "weights"]) + self.target_word["biases"]
                predicted_next_idx = tf.argmax(logits, axis=1)
                last_word = tf.nn.embedding_lookup(
                    self.word_embedding['weights'], predicted_next_idx)
                last_word = last_word + self.word_embedding['biases']
                IDs.append(predicted_next_idx)
        return image_features, IDs

    def train(self, loss, inp_dict):
        self.loss = loss
        self.inp_dict = inp_dict
        saver = tf.train.Saver(max_to_keep=10)
        global_step = tf.Variable(self.current_step, name='global_step', trainable=False)
        starter_learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate, global_step, 100000, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss, global_step=global_step)

        with tf.Session() as sess:
            print "Initializing Training"
            init = tf.global_variables_initializer()
            sess.run(init)
            if self.resume is 1:
                print "Loading Previously Trained Model"
                print self.current_epoch, "Out of", self.nb_epochs, "Completed in previous run."
                try:
                    ckpt_file = "./model/model.ckpt-" + str(self.current_step)
                    saver.restore(sess, ckpt_file)
                    print "Resuming Training"
                except Exception as e:
                    print str(e).split('\n')[0]
                    print "Checkpoints not found"
                    sys.exit(0)

            for epoch in range(self.current_epoch, self.nb_epochs):
                batch_iter = self.get_next_batch()
                for batch_idx in xrange(self.num_batch):
                    batch_features, batch_Ids = batch_iter.next()
                    batch_mask = self.generate_mask(batch_Ids, self.wtoidx)
                    run = [global_step, optimizer, self.loss]
                    feed_dict = self.create_feed_dict(
                        batch_Ids, batch_features, batch_mask)
                    step, _, current_loss = sess.run(run, feed_dict=feed_dict)
                    if step % 100 == 0:
                        print epoch, ": Global Step:", step, "\tLoss: ", current_loss

                print
                print "Epoch: ", epoch, "\tCurrent Loss: ", current_loss
                print "\nSaving Model..\n"
                saver.save(sess, "./model/model.ckpt", global_step=global_step)
                np.save("model/save", (epoch, step))

    def decode(self, path):
        image_features, IDs = self.build_decode_graph()
        saver = tf.train.Saver()
        ckpt_file = "./model/model.ckpt-" + str(self.current_step)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, ckpt_file)
            features = get_features(path)
            features = np.reshape(features, newshape=(1, 1536))
            caption_IDs = sess.run(IDs, feed_dict={image_features: features})
            print self.IDs_to_Words(self.idxtow, caption_IDs)
