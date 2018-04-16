import tensorflow as tf
import math

def sphereface_loss_onehot(x_inputs, y_labels, num_classes, m=4):
    '''
    Args:
        x_inputs = [batch_size features]
        y_labels = [batch_size]
        num_classes:
        m = 
    return 
        logit_final
    '''

    with tf.variable_scope('sphereface_loss_onehot'):
        # [batch_size features]
        x_inputs_shape = x_inputs.get_shape().as_list()

        # [batch_size]
        x_input_norm = tf.sqrt(tf.reduce_sum(tf.square(x_inputs),axis=1))

        # [batch_size features]
        x_inputs_unit = tf.nn.l2_normalize(x_inputs, dim=1)

        # [num_classes features]
        weight = tf.Variable(initial_value=tf.random_normal((num_classes, x_inputs_shape[1])) * tf.sqrt(2 / x_inputs_shape[1]), dtype=tf.float32, name='weight')

        # [num_classes features]
        weight_unit = tf.nn.l2_normalize(weight, dim=1)

        # [batch_size num_classes] cos(theta * m) = link google search: Large-Margin Softmax Loss for Convolutional Neural Networks.pdf 
        cos_theta = tf.matmul(x_inputs_unit, tf.transpose(weight_unit), name='cos_theta')
        cos_theta_square = tf.square(cos_theta)
        cos_theta_pow_4 = tf.pow(cos_theta, 4)
        sign0 = tf.sign(cos_theta)
        sign2 = tf.sign(2 * cos_theta_square - 1)
        sign3 = tf.multiply(sign2, sign0)
        sign4 = 2 * sign0 + sign3 - 3
        cos_m_theta = sign3 * (8 * cos_theta_pow_4 - 8 * cos_theta_square + 1) + sign4

        # [batch_size 1]
        x_input_norm = tf.reshape(x_input_norm, (x_inputs_shape[0], 1))

        # [1 num_classes]
        unit = tf.constant([[1., 1., 1., 1., 1., 1., 1.]])

        # [batch_size num_classes]
        x_input_norm_reshape = tf.matmul(x_input_norm, unit)

        # logit = tf.multiply(cos_theta, x_input_norm) [batch size num_classes] cos(theta) * ||x||
        logit = tf.multiply(cos_theta, x_input_norm_reshape)

        # [batch_size num_class] x [batch_size] = [batch_size num_class] cos(theta * m) * ||x||
        logit_sphereface = tf.multiply(cos_m_theta, x_input_norm_reshape)

        # [batch_size num_classes]
        mask = tf.one_hot(y_labels, depth=num_classes, name='one_hot_mask')

        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        # [batch_size num_classes]
        logit_final = tf.add(tf.multiply(logit, inv_mask), tf.multiply(logit_sphereface, mask), name='arcface_loss_output')
        
        # [batch_size num_classes]
        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_final, labels=y_labels))
        
    return logit_final


def arcface_loss_onehot(x_inputs, y_labels, num_classes, s=64., m=0.5, epsilon=1e-8):
	'''
    Args:
        x_inputs = [batch_size features]
        y_labels = [batch_size]
        num_classes:
        s: scalar for features
        m: 
    return 
        logit_final
    '''

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
		# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_final, labels=y_labels))

	return logit_final



def cosface_loss_onehot(x_inputs, y_labels, num_classes, s=32., m=0.4):
'''
    Args:
        x_inputs = [batch_size features]
        y_labels = [batch_size]
        num_classes:
        s: scalar for features
        m = 
    return 
        logit_final
    '''

    with tf.variable_scope('cosface_loss_onehot'):
        # [batch_size features]
        x_inputs_shape = x_inputs.get_shape().as_list()

        # [batch_size features]
        x_inputs_unit = tf.nn.l2_normalize(x_inputs, dim=1)

        # [batch_size features] ||x|| = s
        x_inputs_s = s * x_inputs_unit

        # [num_classes features]
        weight = tf.Variable(initial_value=tf.random_normal((num_classes, x_inputs_shape[1])) * tf.sqrt(2 / x_inputs_shape[1]), dtype=tf.float32, name='weight')

        # [num_classes features] weight unit
        weight_unit = tf.nn.l2_normalize(weight, dim=1)

        logit = tf.matmul(x_inputs_s, tf.transpose(weight_unit))

        cos_theta = logit / s
        cos_theta_sub_m = cos_theta - m

        logit_cosface = s * (cos_theta_sub_m)

        mask = tf.one_hot(y_labels, depth=num_classes, name='one_hot_mask')

        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        # [batch_size num_classes]
        logit_final = tf.add(tf.multiply(logit, inv_mask), tf.multiply(logit_cosface, mask), name='logit_cosface_mask')

        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_final, labels=y_labels))

    return logit_final

