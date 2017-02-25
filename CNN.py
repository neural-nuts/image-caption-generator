import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os


img_path = "Dataset/flickr30k-images/"
files = sorted(np.array(os.listdir("Dataset/flickr30k-images/")))

batch_size = 10
n_batch = len(files) / batch_size

with open('CNNs/inception_v4.pb', 'rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def)
graph = tf.get_default_graph()
output_layer = graph.get_tensor_by_name("import/InceptionV4/Logits/AvgPool_1a/AvgPool:0")
input_layer = graph.get_tensor_by_name("import/InputImage:0")

'''
def old_load_image(path):
    img = Image.open(path)
    img = np.array(img)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (299, 299, 3))
    return resized_img
'''

def load_image(x, new_h=299, new_w=299):
    image = Image.open(x)
    h, w = image.size
    if image.format != "PNG":
        image = np.asarray(image)/255.0
    else:
        image = np.asarray(image)/255.0
        image = image[:,:,:3]
    '''
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
    '''
    return cv2.resize(image, (new_h, new_w))



def load_next_batch():
    for batch_idx in range(0, len(files), batch_size):
        batch = files[batch_idx:batch_idx + batch_size]
        batch = np.array(map(lambda x: load_image(img_path + x), batch))
        batch = batch.reshape((batch_size, 299, 299, 3))
        yield batch


def forward_pass():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_iter = load_next_batch()
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
            if i % 500 == 0:
                print "Progress:" + str(((i + 1) / float(n_batch) * 100)) + "%\n"
    print "Progress:" + str(((n_batch) / float(n_batch) * 100)) + "%\n"
    print
    print "Saving Features : features.npy\n"
    np.save('Dataset/features', prob)


def init_CNN():
    sess=tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess

def get_features(sess, img):
    image = load_image(img)
    image = image.reshape((1, 299, 299, 3))
    feed_dict = {input_layer: image}
    prob = sess.run(output_layer, feed_dict=feed_dict)
    return prob[0][0]

if __name__ == "__main__":
    print "#Images:", len(files)
    print "Extracting Features"
    forward_pass()
    print "done"
