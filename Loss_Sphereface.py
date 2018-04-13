import tensorflow as tf 

def sphereloss(x_inputs, y_labels, num_classes, fraction=1, scope='Logits', reuse=None, m=4, epsilon=1e-8):
	"""
	x inputs tensor shape=[batch, features_num]
	labels y tensor shape=[batch] each unit belong num_outputs
	batch_size = 256
	features_num = 2
	num_classes = 10
	"""

	# [256, 2]
	x_inputs_shape = x_inputs.get_shape().as_list()

	with tf.variable_scope(name_or_scope=scope):
		# [10, 2]
		weight = tf.Variable(initial_value=tf.random_normal(
			(num_classes, x_inputs_shape[1])) * tf.sqrt(2 / x_inputs_shape[1]),
		dtype=tf.float32, name='weight') # [num_classes, features]

	# [10, 2]
	weight_unit = tf.nn.l2_normalize(weight, dim=1)
	print("weight_unit shape = ", weight_unit.get_shape().as_list())

	# [256] ||x||
	x_inputs_norm = tf.sqrt(tf.reduce_sum(tf.square(x_inputs), axis=1) + epsilon)
	print("x_inputs_norm shape = ", x_inputs_norm.get_shape().as_list())

	# [256, 2] X norm l2
	x_inputs_unit = tf.nn.l2_normalize(x_inputs, dim=1)
	print("x inputs_unit shape = ", x_inputs_unit.get_shape().as_list())

	# [256, 2] [2, 10] = [256, 10] logit
	logit = tf.matmul(x_inputs, tf.transpose(weight_unit))
	print("logit shape = ", logit.get_shape().as_list())

	# [10 2] [256,] = [256 2]
	weight_unit_with_label_in_batch = tf.gather(weight_unit, y_labels)

	# [256] [256 x 2] [256 x 2] after reduce [256] x * w(norm)
	logit_weight_unit = tf.reduce_sum(tf.multiply(x_inputs, weight_unit_with_label_in_batch), axis=1)
	print("logit_weight_unit shape = ", logit_weight_unit.get_shape().as_list())

	# [256]
	cos_theta = tf.reduce_sum(tf.multiply(x_inputs_unit, weight_unit_with_label_in_batch), axis=1)
	print("cos_theta shape = ", cos_theta.get_shape().as_list())

	cos_theta_square = tf.square(cos_theta)
	cos_theta_pow_4 = tf.pow(cos_theta, 4)
	sign0 = tf.sign(cos_theta)
	sign2 = tf.sign(2 * cos_theta_square - 1)
	sign3 = tf.multiply(sign2, sign0)
	sign4 = 2 * sign0 + sign3 - 3
	cos_m_theta = sign3 * (8 * cos_theta_pow_4 - 8 * cos_theta_square + 1) + sign4


	# [256] * [256]  = |x| * cos(theta * m) = [256] cos(m*theta) * ||x|| 
	logit_sphereface = tf.multiply(cos_m_theta, x_inputs_norm)
	print("logit_sphereface shape = ", logit_sphereface.get_shape().as_list())

	# [0, 1, 2, ..., 256]
	index_range = tf.range(start=0, limit=tf.shape(x_inputs, out_type=tf.int64)[0], delta=1, dtype=tf.int64)

	# y_labels shape = [256]
	# x_inputs = [256, 2]
	# [256, 2]
	# [[0, label], [1, label], ..]
	index_labels = tf.stack([index_range, y_labels], axis=1)
	print("index labels shape: ", index_labels.get_shape().as_list())

	# index_labels = [256, 2]
	# logit_ii = [256]
	# logits_inputs = [256]
	# [256, 10]
	index_logits = tf.scatter_nd(index_labels, tf.subtract(logit_sphereface, logit_weight_unit), tf.shape(logit, out_type=tf.int64))
	print("index_logits shape = ", index_logits.get_shape().as_list())

	# [256, 10]
	logit_final = tf.add(logit, index_logits)
	print("logits_final shape = ", logit_final.get_shape().as_list())

	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_labels, logits=logit_final))
	print("loss shape = ", loss.get_shape().as_list())

	return logit_final, loss


def sphereloss_onehot(x_inputs, y_labels, num_classes, fraction=1, scope='Logits', reuse=None, m=4, epsilon=1e-8):
	"""
	x inputs tensor shape=[batch, features_num]
	labels y tensor shape=[batch] each unit belong num_outputs
	batch_size = 256
	features_num = 2
	num_classes = 10
	"""

	with tf.variable_scope('sphereface_loss_onehot'):
		# [batch_size features]
		x_inputs_shape = x_inputs.get_shape().as_list()

		# [batch_size]
		x_input_norm = tf.sqrt(tf.reduce_sum(tf.square(x_inputs),axis=1) + epsilon)

		# [batch_size features]
		x_inputs_unit = tf.nn.l2_normalize(x_inputs, dim=1)

		# [num_classes features]
		weight = tf.Variable(initial_value=tf.random_normal((num_classes, x_inputs_shape[1])) * tf.sqrt(2 / x_inputs_shape[1]), dtype=tf.float32, name='weight')

		# [num_classes features]
		weight_unit = tf.nn.l2_normalize(weight, dim=1)

		# [batch_size num_classes]
		cos_theta = tf.matmul(x_inputs_unit, tf.transpose(weight_unit), name='cos_theta')
		cos_theta_square = tf.square(cos_theta)
		cos_theta_pow_4 = tf.pow(cos_theta, 4)
		sign0 = tf.sign(cos_theta)
		sign2 = tf.sign(2 * cos_theta_square - 1)
		sign3 = tf.multiply(sign2, sign0)
		sign4 = 2 * sign0 + sign3 - 3
		cos_m_theta = sign3 * (8 * cos_theta_pow_4 - 8 * cos_theta_square + 1) + sign4

		x_input_norm = tf.reshape(x_input_norm, (x_inputs_shape[0], 1))
		unit = tf.constant([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

		x_input_norm_reshape = tf.matmul(x_input_norm, unit)
		# logit = tf.multiply(cos_theta, x_input_norm)
		logit = tf.multiply(cos_theta, x_input_norm_reshape)
		# [batch_size num_class] x [batch_size] = [batch_size num_class]
		logit_sphereface = tf.multiply(cos_m_theta, x_input_norm_reshape)

		mask = tf.one_hot(y_labels, depth=num_classes, name='one_hot_mask')

		inv_mask = tf.subtract(1., mask, name='inverse_mask')

		# [batch_size num_classes]
		logit_final = tf.add(tf.multiply(logit, inv_mask), tf.multiply(logit_sphereface, mask), name='arcface_loss_output')

		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_final, labels=y_labels))
		# [batch_size num_classes]
	return logit_final, loss