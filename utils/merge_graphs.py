import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

parser = argparse.ArgumentParser()
parser.add_argument("--encpb", type=str, help="ProtoBuf File of Decoder")
parser.add_argument("--decpb", type=str, help="ProtoBuf File of Encoder")
parser.add_argument(
    "--read_file",
    action="store_true")
args = parser.parse_args()

with open(args.encpb, 'rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
output1="Output_Features:0"
output1=tf.import_graph_def(
    graph_def,
    input_map=None,
    return_elements=[output1],
    name='encoder',
    op_dict=None,
    producer_op_list=None)
graph = tf.get_default_graph()

with open(args.decpb, 'rb') as f:
    fileContent = f.read()
input2="Input_Features"
graph_def2 = tf.GraphDef()
graph_def2.ParseFromString(fileContent)
tf.import_graph_def(
    graph_def2,
    input_map={input2:output1[0]},
    return_elements=None,
    name='decoder',
    op_dict=None,
    producer_op_list=None)
graph = tf.get_default_graph()
tensors_decoder = [n.name for n in tf.get_default_graph().as_graph_def().node]


if args.read_file==1:
    output_node_names = [
        "encoder/Preprocessed_JPG",
        "encoder/Preprocessed_PNG"]
else:
    print "without File I/O"
    output_node_names = []

with open("../model/Decoder/DecoderOutputs.txt", 'r') as f:
    output = f.read()
    prefix = "decoder/"
    output_node_names += [prefix + o for o in output.split('\n')[:-1]]


input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names)
output_graph = "../model/Trained_Graphs/merged_frozen_graph.pb"
if args.read_file == 1:
    output_graph = "../model/Trained_Graphs/merged_frozen_graph_FILE.pb"

with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print "Merged ProtoBuf File Saved:", output_graph
