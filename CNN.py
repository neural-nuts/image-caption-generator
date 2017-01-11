import tensorflow as tf
from PIL import Image
import numpy as np
import skimage
import skimage.io
import skimage.transform
import os

img_path="Dataset/flickr30k-images/"
files=os.listdir("Dataset/flickr30k-images/")

offset=0
batch_size=10
n_batch=len(files)/batch_size

with open('CNNs/inception_v4.pb', 'rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def)
graph = tf.get_default_graph()
tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
print "graph loaded from disk\n\n"

def load_image(path):
    img = skimage.io.imread(path)
    #imshow(Image.open(path,'r'))
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    #crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (299, 299, 3))
    return resized_img


def load_next_batch():
    global offset
    for i, image in enumerate(files[offset*batch_size:offset*batch_size+batch_size]):
        img = skimage.io.imread(img_path+image)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        #crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
        if i is 0:
            resized_img = skimage.transform.resize(crop_img, (299, 299))
        else:
            resized_img = np.append(resized_img, skimage.transform.resize(crop_img, (299, 299)), axis=0)
    batch = resized_img.reshape((batch_size, 299, 299, 3))
    offset+=1
    return batch


def forward_pass():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in xrange(n_batch):
            batch = load_next_batch()
            assert batch.shape == (batch_size, 299, 299, 3)
            feed_dict = {graph.get_tensor_by_name("import/InputImage:0"): batch}
            prob_tensor = graph.get_tensor_by_name("import/InceptionV4/Logits/AvgPool_1a/AvgPool:0")
            if i is 0:
                prob = sess.run(prob_tensor, feed_dict=feed_dict).reshape(batch_size,1536)
            else:
                prob = np.append(prob, sess.run(prob_tensor, feed_dict=feed_dict).reshape(batch_size,1536), axis=0)
            if i%500==0:
                print "Progress:"+ str(((i+1)/float(n_batch)*100))+ "%\n"
    print "Progress:"+ str(((n_batch)/float(n_batch)*100))+ "%\n"
    print
    print "Saving Features : features.np\n"
    np.save('features',prob)

def get_features(path):
    image = load_image(path)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print image.shape
        image = image.reshape((1, 299, 299, 3))
        assert image.shape == (1, 299, 299, 3)
        feed_dict = {graph.get_tensor_by_name("import/InputImage:0"): image}
        prob_tensor = graph.get_tensor_by_name("import/InceptionV4/Logits/AvgPool_1a/AvgPool:0")
        prob = sess.run(prob_tensor, feed_dict=feed_dict)
    return prob

if __name__ == "__main__":
    forward_pass()
    print "done"
