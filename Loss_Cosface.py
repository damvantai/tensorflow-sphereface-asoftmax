import tensorflow as tf
import math

def cosface_loss_onehot(x_inputs, y_labels, num_classes, s=32., m=0.4):
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

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_final, labels=y_labels))

    return logit_final, loss

