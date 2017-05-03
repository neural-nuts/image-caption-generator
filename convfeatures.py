import tensorflow as tf
import numpy as np
import os

batch_size = 10
img_path = "Dataset/flickr30k-images/"
try:
    files = sorted(np.array(os.listdir("Dataset/flickr30k-images/")))
    n_batch = len(files) / batch_size
except:
    pass

with open('ConvNets/inception_v4.pb', 'rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def)
graph = tf.get_default_graph()

input_layer = graph.get_tensor_by_name("import/InputImage:0")
output_layer = graph.get_tensor_by_name(
    "import/InceptionV4/Logits/AvgPool_1a/AvgPool:0")



'''
OLD PRE-PROCESSING MODULES : SLOW
import cv2
from PIL import Image

def old_load_image(x, new_h=299, new_w=299):
    image = Image.open(x)
    h, w = image.size
    if image.format != "PNG":
        image = np.asarray(image)/255.0
    else:
        image = np.asarray(image)/255.0
        image = image[:,:,:3]

    ##To crop or not?
    if w == h:
        resized = cv2.resize(image, (new_h,new_w))
    elif h < w:
        resized = cv2.resize(image, (int(w * float(new_h)/h), new_w))
        crop_length = int((resized.shape[1] - new_h) / 2)
        resized = resized[:,crop_length:resized.shape[1] - crop_length]
    else:
        resized = cv2.resize(image, (new_h, int(h * float(new_w) / w)))
        crop_length = int((resized.shape[0] - new_w) / 2)
        resized = resized[crop_length:resized.shape[0] - crop_length,:]

    return cv2.resize(image, (new_h, new_w))
'''


def build_prepro_graph():
    input_file = tf.placeholder(dtype=tf.string, name="InputFile")
    image_file = tf.read_file(input_file)
    jpg = tf.image.decode_jpeg(image_file, channels=3)
    png = tf.image.decode_png(image_file, channels=3)
    output_jpg = tf.image.resize_images(jpg, [299, 299]) / 255.0
    output_jpg = tf.reshape(
        output_jpg, [
            1, 299, 299, 3], name="Preprocessed_JPG")
    output_png = tf.image.resize_images(png, [299, 299]) / 255.0
    output_png = tf.reshape(
        output_png, [
            1, 299, 299, 3], name="Preprocessed_PNG")
    return input_file, output_jpg, output_png


def load_image(sess, io, image):
    if image.split('.')[-1] == "png":
        return sess.run(io[2], feed_dict={io[0]: image})
    return sess.run(io[1], feed_dict={io[0]: image})


def load_next_batch(sess, io):
    for batch_idx in range(0, len(files), batch_size):
        batch = files[batch_idx:batch_idx + batch_size]
        batch = np.array(
            map(lambda x: load_image(sess, io, img_path + x), batch))
        batch = batch.reshape((batch_size, 299, 299, 3))
        yield batch

def forward_pass(io):
    global output_layer
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_iter = load_next_batch(sess, io)
        for i in xrange(n_batch):
            batch = batch_iter.next()
            assert batch.shape == (batch_size, 299, 299, 3)
            feed_dict = {input_layer: batch}
            if i is 0:
                prob = sess.run(
                    output_layer, feed_dict=feed_dict).reshape(
                    batch_size, 1536)
            else:
                prob = np.append(
                    prob,
                    sess.run(
                        output_layer,
                        feed_dict=feed_dict).reshape(
                        batch_size,
                        1536),
                    axis=0)
            if i % 5 == 0:
                print "Progress:" + str(((i + 1) / float(n_batch) * 100)) + "%\n"
    print "Progress:" + str(((n_batch) / float(n_batch) * 100)) + "%\n"
    print
    print "Saving Features : features.npy\n"
    np.save('Dataset/features', prob)


def get_features(sess, io, img, saveencoder=False):
    global output_layer
    output_layer = tf.reshape(output_layer, [1,1536], name="Output_Features")
    image = load_image(sess, io, img)
    feed_dict = {input_layer: image}
    prob = sess.run(output_layer, feed_dict=feed_dict)

    if saveencoder:
        tensors = [n.name for n in sess.graph.as_graph_def().node]
        with open("model/Encoder/Encoder_Tensors.txt", 'w') as f:
            for t in tensors:
                f.write(t + "\n")
        saver = tf.train.Saver()
        saver.save(sess, "model/Encoder/model.ckpt")
    return prob

if __name__ == "__main__":
    print "#Images:", len(files)
    print "Extracting Features"
    io = build_prepro_graph()
    forward_pass(io)
    print "done"
