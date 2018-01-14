import tensorflow as tf
import numpy as np
import argparse
import os

batch_size = 10
files, input_layer, output_layer = [None]*3

def build_prepro_graph(inception_path):
    global input_layer, output_layer
    with open(inception_path, 'rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    tf.import_graph_def(graph_def)
    graph = tf.get_default_graph()

    input_layer = graph.get_tensor_by_name("import/InputImage:0")
    output_layer = graph.get_tensor_by_name(
        "import/InceptionV4/Logits/AvgPool_1a/AvgPool:0")

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


def load_next_batch(sess, io, img_path):
    for batch_idx in range(0, len(files), batch_size):
        batch = files[batch_idx:batch_idx + batch_size]
        batch = np.array(
            map(lambda x: load_image(sess, io, img_path + x), batch))
        batch = batch.reshape((batch_size, 299, 299, 3))
        yield batch

def forward_pass(io, img_path):
    global output_layer, files
    files = sorted(np.array(os.listdir(img_path)))
    print "#Images:", len(files)
    n_batch = len(files) / batch_size
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_iter = load_next_batch(sess, io, img_path)
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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="A valid path to MSCCOCO/flickr30k images(unzipped)",
        required=True)
    parser.add_argument(
        "--inception_path",
        type=str,
        help="A valid path to inception_v4.pb",
        required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=get_arguments()

    print "Extracting Features"
    io = build_prepro_graph(args.inception_path)
    forward_pass(io, args.data_path)
    print "done"
