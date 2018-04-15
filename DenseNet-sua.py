from __future__ import print_function
import tensorflow as tf
import os
import time
from datetime import timedelta
import numpy as np
import DenseNet_input


class DenseNet:
    def __init__(self, train_data, valid_data=None, test_data=None, num_channel=1, input_size=48, num_class=7,
                 growth_rate=12, reduce_lnr=[150, 225, 300], test_during_train=True,
                 depth=40, total_blocks=3, reduction=1.0, bc_mode=False, keep_prob=1.0, weight_decay=1e-4,
                 init_lnr=0.01, max_to_keep=5,
                 current_save_folder='./save/current/', valid_save_folder='./save/best_valid/',
                 last_save_folder='./save/last/', logs_folder='./summary/', snapshot_test=False):
        '''
        :param train_data: list of pair (image, label) for train model.
        :param valid_data: list of pair (image, label) for validation model.
        :param input_size: 'int', size of input image.
        :param num_class: 'int', number of classes.
        :param growth_rate: 'int', growth rate of DenseNet, default = 12.
        :param depth: 'int', model depth, default = 40.
        :param total_blocks: 'int', number of blocks, default = 3.
        :param reduction: 'float', variable theta in compression.
        :param bc_mode: 'bool', use BC-mode or not.
        :param keep_prob: 'float', keep probability for dropout.
        :param weight_decay: 'float', weight decay for L2-normalization.
        :param weight_init: 'float', stddev for weight initialization.
        '''

        self.num_channel = num_channel
        self.current_save_path = current_save_folder
        self.last_save_path = last_save_folder
        self.valid_save_path = valid_save_folder
        self.logs_folder = logs_folder
        self.max_to_keep = max_to_keep
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.input_size = input_size
        self.num_class = num_class
        self.growth_rate = growth_rate
        self.depth = depth
        self.total_blocks = total_blocks
        self.reduction = reduction
        self.bc_mode = bc_mode
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.init_lnr = init_lnr

        self.reduce_lnr = reduce_lnr
        self.test_during_train = test_during_train

        if not snapshot_test:
            self._input()
            self.inference()
            self.losses()
            self.train_step()
            self.count_trainable_params()
            self.init_session()

    def _input(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.num_channel])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_class])
        self.label = tf.argmax(self.y_, axis=1)
        self.is_training = tf.placeholder(tf.bool)

    def conv2d(self, x, out_filters, filter_size, strides, padding='SAME'):
        in_filters = int(x.get_shape()[-1])
        filter = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                 tf.contrib.layers.variance_scaling_initializer())
        return tf.nn.conv2d(x, filter, [1, strides, strides, 1], padding)

    def avg_pool(self, x, filter_size, strides):
        return tf.nn.avg_pool(x, [1, filter_size, filter_size, 1], [1, strides, strides, 1], padding='VALID')

    def max_pool(self, x, filter_size, strides):
        return tf.nn.max_pool(x, [1, filter_size, filter_size, 1], [1, strides, strides, 1], padding='SAME')

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def batch_norm2(self, _input, is_training):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=is_training,
            updates_collections=None)
        return output

    def batch_norm(self, x, n_out, phase_train=True, scope='bn'):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def composite_function(self, x, out_filters, filter_size=3):
        in_filters = int(x.get_shape()[-1])
        with tf.variable_scope("composite_function"):
            # Batch normalization
            # output = self.batch_norm2(x, self.is_training)
            output = self.batch_norm(x, in_filters, self.is_training)

            # ReLU
            output = tf.nn.relu(output)

            # Conv2d 3x3
            output = self.conv2d(output, out_filters, filter_size, strides=1)
            output = self.dropout(output)

        return output

    def bottleneck(self, x):
        in_filters = int(x.get_shape()[-1])
        out_filters = self.growth_rate * 4
        with tf.variable_scope("bottleneck"):
            # Batch normalization
            # output = self.batch_norm2(x, self.is_training)
            output = self.batch_norm(x, in_filters, self.is_training)

            # ReLU
            output = tf.nn.relu(output)

            # Conv2d 1x1
            output = self.conv2d(output, out_filters, filter_size=1, strides=1, padding='VALID')
            output = self.dropout(output)

        return output

    def add_internal_layer(self, x):
        if self.bc_mode:
            bottleneck = self.bottleneck(x)
            output = self.composite_function(bottleneck, self.growth_rate, filter_size=3)
        else:
            output = self.composite_function(x, self.growth_rate, filter_size=3)

        return tf.concat(axis=3, values=(x, output))

    def add_block(self, input, num_layer):
        output = input
        for layer in range(num_layer):
            with tf.variable_scope("Layer_%d" % layer):
                output = self.add_internal_layer(output)
        return output

    def transition_layer(self, x):
        out_filters = int(int(x.get_shape()[-1]) * self.reduction)
        output = self.composite_function(x, out_filters, filter_size=1)
        output = self.avg_pool(output, 2, 2)
        return output

    # SPHEREFACE LOSS IN LAST_TRANSITION_LAYER
    def last_transition_layer(self, x):
        in_filters = int(x.get_shape()[-1])
        # BN
        # output = self.batch_norm2(x, self.is_training)
        output = self.batch_norm(x, in_filters, self.is_training)

        # ReLU
        output = tf.nn.relu(output)

        # Avg pooling
        filter_size = x.get_shape()[-2]
        output = self.avg_pool(output, filter_size, filter_size)

        # Fully connected
        # total_features = int(output.get_shape()[-1])
        # output = tf.reshape(output, [-1, total_features])
        # W = tf.get_variable('DW', [total_features, self.num_class], tf.float32,
        #                     initializer=tf.contrib.layers.xavier_initializer())
        # b = tf.get_variable('bias', [self.num_class], tf.float32, initializer=tf.constant_initializer(0.0))

        # logits = tf.matmul(output, W) + b

        # return output, logits, total_features



        # SPHEREFACE LOSS
        total_features = int(output.get_shape()[-1])
        output = self.reshape(output, [-1, total_features])
        x_inputs = output
        # batch_size = int(output.get_shape()[0])
        num_classes = self.num_class
        y_labels = self.label

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
        logit_final = tf.add(tf.multiply(logit, inv_mask), tf.multiply(logit_sphereface, mask), name='sphereface_loss_onehot')
        
        # [batch_size num_classes]
        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_final, labels=y_labels))
            
        return x_inputs, logit_final, total_features


    # ARCFACE LOSS IN LAST_TRANSITION_LAYER
"""
    def last_transition_layer(self, x):
        in_filters = int(x.get_shape()[-1])
        # BN
        # output = self.batch_norm2(x, self.is_training)
        output = self.batch_norm(x, in_filters, self.is_training)

        # ReLU
        output = tf.nn.relu(output)

        # Avg pooling
        filter_size = x.get_shape()[-2]
        output = self.avg_pool(output, filter_size, filter_size)

        # Fully connected
        # total_features = int(output.get_shape()[-1])
        # output = tf.reshape(output, [-1, total_features])
        # W = tf.get_variable('DW', [total_features, self.num_class], tf.float32,
        #                     initializer=tf.contrib.layers.xavier_initializer())
        # b = tf.get_variable('bias', [self.num_class], tf.float32, initializer=tf.constant_initializer(0.0))

        # logits = tf.matmul(output, W) + b

        # return output, logits, total_features


        # ARCFACE LOSS
        total_features = int(output.get_shape()[-1])
        output = self.reshape(output, [-1, total_features])
        x_inputs = output
        # batch_size = int(output.get_shape()[0])
        num_classes = self.num_class
        y_labels = self.label


        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m ) * m
        threshold = math.cos(math.pi - m)

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

        # return logit_final, loss
        return x_inputs, logit_final, total_features
"""

    def inference(self):
        self.layer_per_block = int((self.depth - self.total_blocks - 1) / self.total_blocks)

        # First convolutional layer: filter 3 x 3 x (2 * growth rate)
        with tf.variable_scope("Init_conv"):
            output = self.conv2d(self.x, out_filters=2 * self.growth_rate, filter_size=3, strides=1)
            # output = self.max_pool(output, 3, 2)

        # Dense block
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, self.layer_per_block)

                print("Shape after dense block", str(block), ": ", output.get_shape())

                if block != self.total_blocks - 1:
                    with tf.variable_scope("Transition_after_block_%d" % block):
                        output = self.transition_layer(output)

                    print("Shape after transition", str(block), ": ", output.get_shape())

        with tf.variable_scope("Transition_to_classes"):
            self.score, self.logits, self.num_features = self.last_transition_layer(output)
        print("Shape after last block: ", self.logits.get_shape())

        self.prediction = tf.nn.softmax(self.logits)
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def update_centers(self):
        unique_label, unique_idx, unique_count = tf.unique_with_counts(self.label)
        self.appear_times = tf.gather(unique_count, unique_idx) + 1
        self.appear_times = tf.cast(self.appear_times, tf.float32)
        self.appear_times = tf.reshape(self.appear_times, [-1, 1])

        self.update = tf.div(self.diff, self.appear_times)
        self.update_centers_op = tf.scatter_sub(self.centers, self.label, 0.5 * self.update)

    # marginal loss
    def marginal_loss(self, thre = 1.2, slack = 0.3):
        # norm
        A = self.score
        A_l2 = tf.sqrt(tf.reduce_sum(A ** 2, 1, keep_dims=True))
        A_norm = A / A_l2
        # Distance
        r = tf.reduce_sum(A_norm * A_norm, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        distance = r - 2 * tf.matmul(A_norm, tf.transpose(A_norm)) + tf.transpose(r)

        # yij
        y = self.label
        y = tf.one_hot(y, self.num_class)# chuyen thanh one- hot
        r = tf.reduce_sum(y * y, 1)
        r = tf.reshape(r, [-1, 1])
        C = r - 2 * tf.matmul(y, tf.transpose(y)) + tf.transpose(r)

        same = 1-C
        matrix_sub = (slack - same * (thre - distance))
        return tf.reduce_sum(tf.nn.relu( matrix_sub))


    def losses(self):
        self.centers = tf.get_variable('centers', [self.num_class, self.num_features], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0), trainable=False)

        self.centers_batch = tf.gather(self.centers, self.label)


        self.diff = self.centers_batch - self.score

        self.centre_loss = 1.0 * tf.nn.l2_loss(self.diff)
        # margin
        # self.margins = tf.get_variable('margin', [self.num_class, self.num_features], dtype=tf.float32,
        #                                initializer=tf.constant_initializer(0.0), trainable=False)
        #
        # self.margins_batch = tf.gather(self.margins, self.label)
        self.margin_loss = 1e-4 * self.marginal_loss()

        self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_))
        self.total_loss = self.weight_decay * self.l2_loss + self.cross_entropy

    def train_step(self):
        self.learning_rate = tf.placeholder(tf.float32)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9,
                                                     use_nesterov=True).minimize(self.total_loss,
                                                                                 global_step=self.global_step)

        self.train_step2 = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(
            self.centre_loss)
        self.update_centers()

        self.train_step3 = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(
            self.margin_loss)



    def init_session(self):
        if not os.path.isdir(self.current_save_path):
            os.mkdir(self.current_save_path)
            os.mkdir(os.path.join(self.current_save_path, 'current'))

        if not os.path.isdir(self.logs_folder):
            os.mkdir(self.logs_folder)
        self.current_save_path = os.path.join(self.current_save_path, 'current') + '/'
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        if not os.path.isfile(self.current_save_path + 'model.ckpt.index'):
            print('Create new model')
            self.sess.run(tf.global_variables_initializer())
            print('OK')
        else:
            print('Restoring existed model')
            self.saver.restore(self.sess, self.current_save_path + 'model.ckpt')
            print('OK')
            print(self.global_step.eval())
        writer = tf.summary.FileWriter
        self.summary_writer = writer(self.logs_folder)

    def count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix):
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def train(self, num_epoch, batch_size=512):
        train_img = []
        train_label = []
        for i in range(len(self.train_data)):
            train_img.append(self.train_data[i][0])
            train_label.append(self.train_data[i][1])

        num_batch = int(len(train_img) // batch_size)
        current_epoch = int(self.global_step.eval() / num_batch)

        lnr = self.init_lnr

        for epoch in range(current_epoch + 1, num_epoch + 1):
            current_step = int(self.global_step.eval())
            print('Epoch:', str(epoch))
            np.random.shuffle(self.train_data)
            start_time = time.time()
            train_img = []
            train_label = []
            for i in range(len(self.train_data)):
                train_img.append(self.train_data[i][0])
                train_label.append(self.train_data[i][1])

            for i in range(len(self.reduce_lnr)):
                if self.reduce_lnr[i] <= epoch:
                    lnr = self.init_lnr * (0.1 ** (i + 1))

            print("Learning rate: %f" % lnr)
            sum_loss = []
            sum_l2 = []
            sum_acc = []
            sum_centre_loss = []
            sum_margin_loss = []
            for batch in range(num_batch):
                top = batch * batch_size
                bot = min((batch + 1) * batch_size, len(self.train_data))
                batch_img = np.asarray(train_img[top:bot])
                batch_label = np.asarray(train_label[top:bot])

                batch_img = DenseNet_input.augmentation(batch_img, self.input_size)

                ttl, l2l, ctl, mgl, _, __, ___,____, acc = self.sess.run(
                    [self.total_loss, self.l2_loss, self.centre_loss, self.margin_loss , self.train_step, self.train_step2,self.train_step3, self.update_centers_op,
                     self.accuracy], feed_dict={self.x: batch_img, self.y_: batch_label, self.is_training: True,
                                                self.learning_rate: lnr})
                print('Training on batch %s / %s' % (str(batch + 1), str(num_batch)), end='\r')
                sum_loss.append(ttl)
                sum_acc.append(acc)
                sum_l2.append(l2l)
                sum_centre_loss.append(ctl)
                sum_margin_loss.append(mgl)
            time_per_epoch = time.time() - start_time
            mean_loss = np.mean(sum_loss)
            mean_acc = np.mean(sum_acc)
            mean_l2 = np.mean(sum_l2)
            mean_centre_loss = np.mean(sum_centre_loss)
            mean_margin_loss = np.mean(sum_margin_loss)
            print('\nTraining time: ', str(timedelta(seconds=time_per_epoch)))
            print('Training loss: %f' % mean_loss)
            print('L2 loss: %f' % mean_l2)
            print('Centre loss: %f' % mean_centre_loss)
            print('Margin loss: %f' % mean_margin_loss)
            print('Train accuracy: %f' % mean_acc)
            self.saver.save(self.sess, save_path=self.current_save_path + 'model' + str(epoch) + '.ckpt')
            self.saver.save(self.sess, save_path=self.current_save_path + 'model.ckpt')
            self.log_loss_accuracy(loss=mean_loss, accuracy=mean_acc, epoch=epoch, prefix='train')

            mean_loss, mean_acc = self.valid(batch_size=batch_size)
            self.log_loss_accuracy(loss=mean_loss, accuracy=mean_acc, epoch=epoch, prefix='valid')
            if mean_acc >= 0.67:
                self.saver.save(self.sess, save_path=self.valid_save_path + 'model' + str(round(mean_acc, 4)) + '.ckpt')

            if self.test_during_train:
                mean_loss, mean_acc, _, _ = self.test(batch_size=batch_size)
                self.log_loss_accuracy(loss=mean_loss, accuracy=mean_acc, epoch=epoch, prefix='test')

    def test(self, batch_size=1, save_file=None):
        print('\n')
        if save_file is not None:
            self.sess.close()
            self.sess = tf.InteractiveSession()
            print("Restoring session from %s for testing" % save_file)
            self.saver.restore(self.sess, self.current_save_path + save_file)
            print("OK")
        all_score = []
        all_label = []
        test_img = []
        test_label = []
        for i in range(len(self.test_data)):
            test_img.append(self.test_data[i][0])
            test_label.append(self.test_data[i][1])

        num_batch = int(len(test_img) // batch_size)
        sum_loss = []
        sum_acc = []
        for batch in range(num_batch):
            top = batch * batch_size
            bot = min((batch + 1) * batch_size, len(self.test_data))
            batch_img = np.asarray(test_img[top:bot])
            batch_label = np.asarray(test_label[top:bot])
            batch_img = DenseNet_input.augmentation(batch_img, self.input_size, testing=True)
            logits = self.sess.run([self.prediction],
                                   feed_dict={self.x: batch_img, self.y_: batch_label, self.is_training: False})
            logits = np.asarray(logits)
            for i in range(logits.shape[1]):
                all_score.append(logits[0][i])
                all_label.append(batch_label[i])

            ttl, l2l, acc = self.sess.run([self.total_loss, self.l2_loss, self.accuracy],
                                          feed_dict={self.x: batch_img, self.y_: batch_label,
                                                     self.is_training: False})
            print('Testing on batch %s / %s' % (str(batch + 1), str(num_batch)), end='\r')
            sum_loss.append(ttl)
            sum_acc.append(acc)
        mean_loss = np.mean(sum_loss)
        mean_acc = np.mean(sum_acc)
        print('\nTest loss: %f' % mean_loss)
        print('Test accuracy: %f' % mean_acc)
        return mean_loss, mean_acc, all_score, all_label

    def test_snapshot_ensemble(self, batch_size=1, save_file=None):
        test_img = []
        test_label = []
        for i in range(len(self.test_data)):
            test_img.append(self.test_data[i][0])
            test_label.append(self.test_data[i][1])

        num_batch = int(len(test_img) // batch_size)
        prediction = np.zeros([len(test_img), self.num_class])
        num_saves = len(save_file)
        all_label = []
        for i in range(num_saves):
            print("Checkpoint %d" % save_file[i])
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            self._input()
            self.inference()
            self.losses()
            self.count_trainable_params()

            saver = tf.train.Saver()
            saver.restore(sess, self.valid_save_path + "model{}.ckpt".format(save_file[i]))
            for batch in range(num_batch):
                top = batch * batch_size
                bot = min((batch + 1) * batch_size, len(self.test_data))
                batch_img = np.asarray(test_img[top:bot])
                batch_label = np.asarray(test_label[top:bot])

                if i == 0:
                    all_label.append(batch_label[i])

                batch_img = DenseNet_input.augmentation(batch_img, self.input_size, testing=True)
                logits = sess.run([self.logits],
                                  feed_dict={self.x: batch_img, self.y_: batch_label, self.is_training: False})
                logits = np.asarray(logits)
                prediction[top] += logits[0][0]

        correct_pred = np.equal(np.argmax(prediction, 1), np.argmax(np.asarray(test_label), 1))
        accuracy = np.mean(correct_pred.astype(int))
        print("Snapshot test accuracy: {}".format(accuracy))
        return prediction, all_label

    def valid(self, batch_size=1):
        print('\n')
        valid_img = []
        valid_label = []
        for i in range(len(self.valid_data)):
            valid_img.append(self.valid_data[i][0])
            valid_label.append(self.valid_data[i][1])

        num_batch = int(len(valid_img) // batch_size)
        sum_loss = []
        sum_acc = []
        for batch in range(num_batch):
            top = batch * batch_size
            bot = min((batch + 1) * batch_size, len(self.valid_data))
            batch_img = np.asarray(valid_img[top:bot])
            batch_label = np.asarray(valid_label[top:bot])
            batch_img = DenseNet_input.augmentation(batch_img, self.input_size, testing=True)
            ttl, l2l, acc = self.sess.run([self.total_loss, self.l2_loss, self.accuracy],
                                          feed_dict={self.x: batch_img, self.y_: batch_label,
                                                     self.is_training: False})
            print('Validating on batch %s / %s' % (str(batch + 1), str(num_batch)), end='\r')
            sum_loss.append(ttl)
            sum_acc.append(acc)
        mean_loss = np.mean(sum_loss)
        mean_acc = np.mean(sum_acc)
        print('\nValid loss: %f' % mean_loss)
        print('Valid accuracy: %f' % mean_acc)
        return mean_loss, mean_acc

    def get_score(self, batch_size=1, save_file=None):
        test_img = []
        test_label = []
        for i in range(len(self.test_data)):
            test_img.append(self.test_data[i][0])
            test_label.append(self.test_data[i][1])

        num_batch = int(len(test_img) // batch_size)
        self._input()
        self.inference()
        self.losses()
        prediction = np.zeros([len(test_img), self.num_features])
        num_saves = len(save_file)
        all_label = []
        for i in range(num_saves):
            print("Checkpoint %d" % save_file[i])
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            self._input()
            self.inference()
            self.losses()
            self.count_trainable_params()

            saver = tf.train.Saver()
            saver.restore(sess, self.valid_save_path + "model{}.ckpt".format(save_file[i]))
            for batch in range(num_batch):
                top = batch * batch_size
                bot = min((batch + 1) * batch_size, len(self.test_data))
                batch_img = np.asarray(test_img[top:bot])
                batch_label = np.asarray(test_label[top:bot])

                if i == 0:
                    all_label.append(batch_label[i])

                batch_img = DenseNet_input.augmentation(batch_img, self.input_size, testing=True)
                logits = sess.run([self.score],
                                  feed_dict={self.x: batch_img, self.y_: batch_label, self.is_training: False})
                logits = np.asarray(logits)
                prediction[top] += logits[0][0]
        return prediction, all_label
