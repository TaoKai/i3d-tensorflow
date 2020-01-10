import os, sys
import numpy as np
import tensorflow as tf
from i3d import buildMyTestNet
from data_prepare import MyData
from tensorflow.python.framework import graph_util

def load_pb(path):
    with tf.Graph().as_default():
        pb_graph_def = tf.GraphDef()
        with open(path, "rb") as f:
            pb_graph_def.ParseFromString(f.read())
            tf.import_graph_def(pb_graph_def, name='')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        inputs = sess.graph.get_tensor_by_name("inputs:0")
        predictions = sess.graph.get_tensor_by_name("MyI3D/Softmax:0")
        return sess, inputs, predictions

def save_pb(sess, names, out_path):
    pb_graph = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=tf.get_default_graph().as_graph_def(),
        output_node_names=names)
    with tf.gfile.GFile(out_path, "wb") as f:
        f.write(pb_graph.SerializeToString())
        print(names)
        print("%d ops in the final graph." % len(pb_graph.node))

def evalByPB():
    data_flow = MyData('./data', batch_size=16)
    sess, inputs, pred = load_pb('./testPB.pb')
    test_num = data_flow.get_test_num()
    total = 0
    total_pass = 0
    for i in range(test_num):
        batch_data = data_flow.next_test()
        imgs = batch_data[0]
        labels = batch_data[1]
        names = batch_data[2]
        probs = sess.run(pred, {inputs:imgs})
        pred_labels = np.argmax(probs, axis=1)
        acc_vec = (pred_labels==labels).astype(np.int32)
        total += labels.shape[0]
        total_pass += np.sum(acc_vec)
        print('evaluating the test set ...', total_pass, '/', total)
    print('the accuracy is', total_pass*1.0/total)


def myEval():
    data_path = './data'
    data_flow = MyData(data_path, batch_size=16)
    restore_path = './models/best_model.ckpt'
    sess = tf.Session()
    pred, inputs = buildMyTestNet(sess, is_training=False, restore_path=restore_path)
    #save_pb(sess, ['inputs', 'MyI3D/Softmax'], './testPB.pb')
    test_num = data_flow.get_test_num()
    total = 0
    total_pass = 0
    for i in range(test_num):
        batch_data = data_flow.next_test()
        imgs = batch_data[0]
        labels = batch_data[1]
        names = batch_data[2]
        probs = sess.run(pred, {inputs:imgs})
        pred_labels = np.argmax(probs, axis=1)
        acc_vec = (pred_labels==labels).astype(np.int32)
        total += labels.shape[0]
        total_pass += np.sum(acc_vec)
        print('evaluating the test set ...', total_pass, '/', total)
    print('the accuracy is', total_pass*1.0/total)
        
if __name__ == "__main__":
    #evalByPB()
    myEval()
