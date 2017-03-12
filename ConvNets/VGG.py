import tensorflow as tf
from PIL import Image
import numpy as np
import skimage
import skimage.io
import skimage.transform
import os
import matplotlib.pyplot as plt

img_path = "Dataset/flickr30k-images/"
files = sorted(np.array(os.listdir("Dataset/flickr30k-images/")))

batch_size = 10
n_batch = len(files) / batch_size

with open('CNNs/vgg16.pb', 'rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def)
graph = tf.get_default_graph()
tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
print "graph loaded from disk\n\n"


def load_image(path):
    img = skimage.io.imread(path)
    #fig = plt.figure()
    # a=fig.add_subplot(1,2,1)
    # x=skimage.io.imshow(img)
    # a.set_title('Before')
    # print img.shape
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224, 3))
    # a=fig.add_subplot(1,2,2)
    # skimage.io.imshow(resized_img)
    # x.set_clim(0.0,0.7)
    # a.set_title('After')
    # plt.show()
    return resized_img


def load_next_batch():
    for batch_idx in range(0, len(files), batch_size):
        batch = files[batch_idx:batch_idx + batch_size]
        batch = np.array(map(lambda x: load_image(img_path + x), batch))
        batch = batch.reshape((batch_size, 224, 224, 3))
        yield batch


def forward_pass():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_iter = load_next_batch()
        for i in xrange(n_batch):
            batch = batch_iter.next()
            assert batch.shape == (batch_size, 224, 224, 3)
            feed_dict = {graph.get_tensor_by_name(
                "import/images:0"): batch}
            prob_tensor = graph.get_tensor_by_name(
                "import/conv5_3/Relu:0")
            if i is 40:
                break
            if i is 0:
                prob = sess.run(prob_tensor, feed_dict=feed_dict).reshape(-1, 196, 512).swapaxes(1,2)
            else:
                prob = np.append(prob, sess.run(prob_tensor, feed_dict=feed_dict).reshape(-1, 196, 512).swapaxes(1,2), axis=0)
            if i % 500 == 0:
                print "Progress:" + str(((i + 1) / float(n_batch) * 100)) + "%\n"
    print "Progress:" + str(((n_batch) / float(n_batch) * 100)) + "%\n"
    print
    print "Saving Features : featuresVGG.npy\n"
    np.save('Dataset/featuresVGG', prob)


def get_features(path):
    image = load_image(path)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        image = image.reshape((1, 224, 224, 3))
        assert image.shape == (1, 224, 224, 3)
        feed_dict = {graph.get_tensor_by_name("import/InputImage:0"): image}
        prob_tensor = graph.get_tensor_by_name(
            "import/InceptionV4/Logits/AvgPool_1a/AvgPool:0")
        prob = sess.run(prob_tensor, feed_dict=feed_dict)
    return prob

if __name__ == "__main__":
    print "#Images:", len(files)
    print "Extracting Features"
    forward_pass()
    print "done"
