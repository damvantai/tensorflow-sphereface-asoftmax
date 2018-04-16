import tensorflow as tf
import math

# def arcface_loss(x_inputs, y_labels, num_classes, s=64.0, m=0.5, epsilon=1e-8):
# 	"""
# 	Args:
# 		x_inputs: [batch_size, features_num]
# 		y_labels: [batch_size]
# 		num_classes:
# 		s: scalar
# 		m: angular margin
# 	Return:
# 		loss, logit
# 	"""

# 	cos_m = math.cos(m)
# 	sin_m = math.sin(m)
# 	mm = math.sin(math.pi - m) * m
# 	threshold = math.cos(math.pi - m)

# 	x_inputs_shape = x_inputs.get_shape().as_list()

# 	with tf.variable_scope(name_or_scope="weight"):
# 		# [num_classes features]
# 		weight = tf.Variable(initial_value=tf.random_normal((
# 			num_classes, x_inputs_shape[1])) * tf.sqrt(2 / x_inputs_shape[1]),
# 		dtype=tf.float32, name='weight')

# 	# [num_classes features]
# 	weight_unit = tf.nn.l2_normalize(weight, dim=1)
# 	print("weight_unit shape = ", weight_unit.get_shape().as_list())

# 	# # [batch_size]
# 	# x_inputs_norm = tf.sqrt(tf.reduce_sum(tf.square(x_inputs), axis=1) + epsilon)

# 	# [batch_size, features]
# 	x_inputs_unit = tf.nn.l2_normalize(x_inputs, dim=1)
# 	print("x_inputs_unit shape = ", x_inputs_unit.get_shape().as_list())

# 	# [batch_size num_classes]
# 	logits = s * tf.matmul(x_inputs_unit, tf.transpose(weight_unit))
# 	print("logits shape = ", logits.get_shape().as_list())

# 	# [batch_size features]
# 	weight_unit_with_label_in_batch = tf.gather(weight_unit, y_labels)

	
# 	# logit_weight_unit_x_unit = tf.reduce_sum(tf.multiply(x_inputs_unit, weight_unit_with_label_in_batch), axis=1)
# 	# print("logit_weight_unit_x_unit shape = ", logit_weight_unit_x_unit.get_shape().as_list())

# 	# [batch_size] logit_weight_unit_x_unit
# 	cos_theta = tf.reduce_sum(tf.multiply(x_inputs_unit, weight_unit_with_label_in_batch), axis=1)

# 	# [batch_size]
# 	logit_weight_unit_x_unit = s * cos_theta

# 	cos_theta_square = tf.square(cos_theta, name="cos_theta_square")
# 	sin_theta_square = tf.subtract(1., cos_theta_square)
# 	sin_theta = tf.sqrt(sin_theta_square, name='sin_theta_square')

# 	# [batch_size]
# 	cos_theta_plus_m = tf.subtract(tf.multiply(cos_theta, cos_m), tf.multiply(sin_theta, sin_m), name='cos_theta_plus_m')
	
# 	# [batch_size]
# 	logits_arcface = s * cos_theta_plus_m
# 	print("logit_arcface shape = ", logits_arcface.get_shape().as_list())

# 	# [0, 1, 2.. , batch_size]
# 	index_range = tf.range(start=0, limit=tf.shape(x_inputs, out_type=tf.int64)[0], delta=1, dtype=tf.int64)

# 	# [[0, label], [1, label], ... [batch_size, label]]
# 	index_labels = tf.stack([index_range, y_labels], axis=1)
# 	print("index labels shape: ", index_labels.get_shape().as_list())

# 	index_logits = tf.scatter_nd(index_labels, tf.subtract(logits_arcface,logit_weight_unit_x_unit), tf.shape(logits, out_type=tf.int64))
# 	print("index_logit shape = ", index_logits.get_shape
# 	().as_list())

# 	# [batch_size num_classes]
# 	logit_final = tf.add(logits, index_logits)
# 	print("logits_final shape = ", logit_final.get_shape().as_list())

# 	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_labels, logits=logit_final))

# 	return logit_final, loss


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
