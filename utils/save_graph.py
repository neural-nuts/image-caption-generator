import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(mode, read_file ,model_folder):

    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = "../model/Trained_Graphs/" + mode + "_frozen_model.pb"
    if mode == 'encoder':
        if read_file:
            output_node_names = [
                "Preprocessed_JPG",
                "Preprocessed_PNG",
                "Output_Features"]
        else:
            print "without file I/O"
            output_node_names = ["Output_Features"]
    if mode == 'decoder':
        with open("../model/Decoder/DecoderOutputs.txt", 'r') as f:
            output_node_names = f.read()
            output_node_names = output_node_names.split('\n')[:-1]

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names)

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print "ProtoBuf File Saved:", output_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["encoder","decoder"])
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Model folder to export")
    parser.add_argument(
        "--read_file",
        action="store_true")
    args = parser.parse_args()
    freeze_graph(args.mode, args.read_file, args.model_folder)
