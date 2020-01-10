""" This implementation based on naive tensorflow framework
Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.
The model is introduced in:
  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from net_utils import unit3D

def I3D(inputs,
        num_classes=400,
        is_training=True,
        final_endpoint='Predictions',
        data_format='NHWC',
        dropout_keep_prob=0.5,
        min_depth=16,
        depth_multiplier=1.0,
        scope=None):
	end_points = {}
	if depth_multiplier <= 0:
		raise ValueError('depth_multiplier is not greater than zero.')
	depth = lambda d: max(int(d * depth_multiplier), min_depth)

	concat_axis = 2 if data_format == 'NCHW' else -1
	with tf.variable_scope(scope, 'I3D', [inputs]):
		end_point = 'Conv3d_1a_7x7x7'
		net = unit3D(inputs, depth(64), [7,7,7], 2, is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_2a_1x3x3'
		net = tf.nn.max_pool3d(net, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME', name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Conv3d_2b_1x1x1'
		net = unit3D(net, depth(64), [1, 1, 1], is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Conv3d_2c_3x3x3'
		net = unit3D(net, depth(192), [3, 3, 3], is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_3a_1x3x3'
		net = tf.nn.max_pool3d(net, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME', name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_3b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(96), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(16), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(32), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, [1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
				                            padding='SAME', name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)

		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_3c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(192), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
		net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_4a_3x3x3'
		net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                               padding='SAME', name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_4b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(192), kernel_shape=[1, 1, 1],
                           		  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(96), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(208), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(16), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(48), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
		net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_4c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(160), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(112), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(224), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(24), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(64), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_4d'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(256), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(24), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(64), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')

			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net

		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_4e'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(112), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(144), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(288), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(64), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')

			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_4f'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(256), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(160), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(320), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_5a_2x2x2'
		net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
							   padding='SAME', name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_5b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(256), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(160), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(320), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0a_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_5c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(384), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(192), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(384), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(48), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Logits'
		with tf.variable_scope(end_point):
			net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
								   strides=[1, 1, 1, 1, 1], padding='VALID')
			net = tf.nn.dropout(net, dropout_keep_prob)
			logits = unit3D(net, num_classes,
							kernel_shape=[1, 1, 1],
							activation_fn=None,
							is_training=is_training,
							use_batch_norm=False,
							use_bias=True,
							name='Conv3d_0c_1x1x1')
			logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
			averaged_logits = tf.reduce_mean(logits, axis=1)
		end_points[end_point] = averaged_logits
		if end_point == final_endpoint: return averaged_logits, end_points

		end_point = 'Predictions'
		predictions = tf.nn.softmax(averaged_logits)
		end_points[end_point] = predictions
		if end_point == final_endpoint: return predictions, end_points


def buildMyNet(sess, num_classes=101, is_training=True, dropout_keep_prob=0.5):
	modelPath = './weights/model.ckpt'
	inputs = tf.placeholder(dtype=tf.float32, shape=[16, None, 224, 224, 3])
	drop_out = tf.placeholder(dtype=tf.float32, shape=None)
	net, end_points = I3D(inputs, final_endpoint='Mixed_5c', is_training=is_training)
	with tf.variable_scope('MyI3D'):
		end_point = 'Logits'
		with tf.variable_scope(end_point):
			net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
								   strides=[1, 1, 1, 1, 1], padding='VALID')
			net = tf.nn.dropout(net, drop_out)
			logits = unit3D(net, num_classes,
							kernel_shape=[1, 1, 1],
							activation_fn=None,
							is_training=is_training,
							use_batch_norm=False,
							use_bias=True,
							name='Conv3d_0c_1x1x1')
			logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
			averaged_logits = tf.reduce_mean(logits, axis=1)
		end_points[end_point] = averaged_logits
		end_point = 'Predictions'
		predictions = tf.nn.softmax(averaged_logits)
		end_points[end_point] = predictions
		labels = tf.placeholder(tf.int32, (None,), name='labels')
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels, name='loss')
		loss = tf.reduce_mean(loss)
		opt = tf.train.AdamOptimizer(learning_rate=0.0003)
		grads = opt.compute_gradients(loss)
		opt_apply = opt.apply_gradients(grads)

	sess.run(tf.global_variables_initializer())
	for k, v in end_points.items():
		print(k, v)
	# show all vars
	print('----------------variables to restore--------------------')
	allVars = tf.global_variables()
	feed_dict = {v.name[:-2]:v for v in allVars if 'MyI3D' not in v.name and 'BatchNorm/gamma' not in v.name}
	for k, v in feed_dict.items():
		print(k, v)
	saver = tf.train.Saver(var_list=feed_dict)
	saver.restore(sess, modelPath)
	
	return predictions, inputs, drop_out, loss, labels, opt_apply
		

def buildMyTestNet(sess, num_classes=101, is_training=True, dropout_keep_prob=1.0, restore_path=None):
	modelPath = restore_path
	inputs = tf.placeholder(dtype=tf.float32, shape=[None, 64, 224, 224, 3], name='inputs')
	net, end_points = I3D(inputs, final_endpoint='Mixed_5c', is_training=is_training)
	with tf.variable_scope('MyI3D'):
		end_point = 'Logits'
		with tf.variable_scope(end_point):
			net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
								   strides=[1, 1, 1, 1, 1], padding='VALID')
			net = tf.nn.dropout(net, dropout_keep_prob)
			logits = unit3D(net, num_classes,
							kernel_shape=[1, 1, 1],
							activation_fn=None,
							is_training=is_training,
							use_batch_norm=False,
							use_bias=True,
							name='Conv3d_0c_1x1x1')
			logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
			averaged_logits = tf.reduce_mean(logits, axis=1)
		end_points[end_point] = averaged_logits
		end_point = 'Predictions'
		predictions = tf.nn.softmax(averaged_logits)
		end_points[end_point] = predictions

	sess.run(tf.global_variables_initializer())
	for k, v in end_points.items():
		print(k, v)
	# show all vars
	print('----------------variables to restore--------------------')
	allVars = tf.global_variables()
	feed_dict = {v.name[:-2]:v for v in allVars}
	for k, v in feed_dict.items():
		print(k, v)
	saver = tf.train.Saver(var_list=feed_dict)
	if modelPath is not None:
		saver.restore(sess, modelPath)
	
	return predictions, inputs


if __name__ == '__main__':
	# inputs: [batch_size, num_frames, h, w, c], outputs: [batch_size, num_classes]
	sess = tf.Session()
	pred, inputs, drop_out, loss, labels, opt = buildMyNet(sess)
	# pred2, inputs2 = buildMyTestNet(sess)
	
	for _ in range(1):
		fake_v = np.random.random([8, 64, 224, 224, 3])*2-1
		fake_l = np.random.randint(0, 100, size=(8,), dtype=np.int32)
		res = sess.run([loss, pred], feed_dict={inputs:fake_v, drop_out:0.5, labels:fake_l})
		print(res)
	

