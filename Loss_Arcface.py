import tensorflow as tf
import math


def arcface_loss_onehot(x_inputs, y_labels, num_classes, s=64., m=0.5, epsilon=1e-8):
	"""
	x inputs tensor shape=[batch, features_num]
	labels y tensor shape=[batch] each unit belong num_outputs
	batch_size = 256
	features_num = 2
	num_classes = 10
	"""

	cos_m = math.cos(m)
	sin_m = math.sin(m)
	mm = math.sin(math.pi - m ) * m
	threshold = math.cos(math.pi - m)

	with tf.variable_scope('arcface_loss_onehot'):
		# [batch_size features]
		x_inputs_shape = x_inputs.get_shape().as_list()

		# [batch_size] X_norm
		# x_input_norm = tf.sqrt(tf.reduce_sum(tf.square(x_inputs),axis=1) + epsilon)

		# [batch_size features] ||X||
		x_inputs_unit = tf.nn.l2_normalize(x_inputs, dim=1)
		# ||X|| * s
		x_inputs_unit_s = s * x_inputs_unit

		# [num_classes features] W
		weight = tf.Variable(initial_value=tf.random_normal((num_classes, x_inputs_shape[1])) * tf.sqrt(2 / x_inputs_shape[1]), dtype=tf.float32, name='weight')

		# [num_classes features] ||W||
		weight_unit = tf.nn.l2_normalize(weight, dim=1)

		logit = tf.matmul(x_inputs_unit_s, tf.transpose(weight_unit))

		# [batch_size num_classes]
		# cos(theta + m) = cos(theta)*cos(m) - sin(theta)sin(m)
		cos_theta = logit / s
		cos_theta_square = tf.square(cos_theta)
		sin_theta_square = tf.subtract(1., cos_theta_square)
		sin_theta = tf.sqrt(sin_theta_square)
		cos_theta_m = tf.subtract(tf.multiply(cos_theta, cos_m), tf.multiply(sin_theta, sin_m))

		# [batch_size num_classes] s * cos(theta + m)
		logit_arcface = s * cos_theta_m

		# 
		cond_v = cos_theta - threshold
		cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
		keep_val = logit - s*mm
		
		# if cond true logit_arcface else keep_val
		logit_arcface = tf.where(cond, logit_arcface, keep_val)
		
		# [batch_size num_classes] one hot of y_labels
		mask = tf.one_hot(y_labels, depth=num_classes, name='one_hot_mask')

		inv_mask = tf.subtract(1., mask, name='inverse_mask')

		# [batch_size num_classes]
		logit_final = tf.add(tf.multiply(logit, inv_mask), tf.multiply(logit_arcface, mask), name='arcface_loss_output')

		# loss
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_final, labels=y_labels))

	return logit_final, loss
