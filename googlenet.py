import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

import utils

class GOOGLENET():

    def __init__(self, config, inputs):

        self.config = config

        self.image_size = inputs.image_size
        self.class_num = inputs.class_num

        self.model_name = "GoogLeNet-has"

        self.ox = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='ox')

        self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='x')
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')
        self.bbox = tf.placeholder(tf.int64, shape=[None, 4])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.build_model()
        self.build_loss_and_optimizer()
        self.merge_summary()

    def build_model(self):

        net = self.x
        print(net.shape)

        with tf.variable_scope(self.model_name):

            with slim.arg_scope([slim.conv2d], #, slim.fully_connected, slim.max_pool2d]
                                # activation_fn=tf.nn.elu,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.001),
                                biases_initializer=tf.constant_initializer(0.1),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': self.is_training,
                                                   'decay': 0.95,
                                                   'center': True,
                                                   'scale': True,
                                                   'activation_fn': tf.nn.elu,
                                                   'updates_collections': None},
                                ):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    stride=1, padding='SAME'):

                    net = slim.conv2d(net, 64, [3, 3], stride=2, scope='Conv2d_1a_3x3') # net = slim.conv2d(net, 64, [7, 7], stride=2, scope='Conv2d_1a_7x7')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_2a_3x3')
                    net = slim.conv2d(net, 64, [1, 1], scope='Conv2d_2b_1x1')
                    net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_2c_3x3')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
                    print(net.shape)

                    with tf.variable_scope('Mixed_3b'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], padding='SAME', scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                            print(branch_3.shape)
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                    print(net.shape)

                    with tf.variable_scope('Mixed_3c'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                    print(net.shape)

                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_4a_3x3')
                    print(net.shape)

                    with tf.variable_scope('Mixed_4b'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                    print(net.shape)

                    with tf.variable_scope('Mixed_4c'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                    print(net.shape)

                    with tf.variable_scope('Mixed_4d'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                    print(net.shape)

                    with tf.variable_scope('Mixed_4e'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                    print(net.shape)

                    # googlenet-gap
                    net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                    net = slim.conv2d(net, 1024, [3, 3], activation_fn=None, normalizer_fn=None, scope='conv7')

                    # net = slim.batch_norm(net, scope='bn7')  # fixme: cam을 위한 conv 구할 때 batch_norm, activation을 해야하나
                    self.conv = net  # later use this for cam

                    print(net.shape)

                    # gap
                    net = tf.reduce_mean(net, [1, 2], name='global_pool')  #keep_dims
                    self.gap = net

                    # cam
                    with tf.variable_scope("gap"):
                        gap_w = tf.get_variable(
                            "W",
                            shape=[1024, self.class_num],
                            initializer=tf.random_normal_initializer(0., 0.01))

                    # classification
                    self.logits = tf.matmul(self.gap, gap_w)



    def get_classmap(self, label, conv):
        conv_resized = tf.image.resize_bilinear(conv, [self.image_size, self.image_size])
        with tf.variable_scope(self.model_name):
            with tf.variable_scope("gap", reuse=True):
                label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
                label_w = tf.reshape(label_w, [-1, 1024, 1])  # [batch_size, 1024, 1]

        conv_resized = tf.reshape(conv_resized,
                                   [-1, self.image_size * self.image_size, 1024])  # [batch_size, 224*224, 1024]

        classmap = tf.matmul(conv_resized, label_w)
        classmap = tf.reshape(classmap, [-1, self.image_size, self.image_size])
        return classmap

    def build_loss_and_optimizer(self):

        #
        # Classification Loss - SparseSoftmaxCrossEntropyLoss, WeightedSigmoidClassificationLoss
        #
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.cross_entropy_loss = tf.reduce_mean(self.cross_entropy)
        self.cls_loss = self.cross_entropy_loss

        self.pred_class = tf.argmax(self.logits, axis=1)

        self.corrections = tf.equal(self.y, self.pred_class)
        self.cls_accuracy = tf.reduce_sum(tf.cast(self.corrections, tf.int32)) / tf.shape(self.y)[0]
        # self.accuracy = tf.metrics.accuracy(labels=self.y,
        #                                     predictions=tf.argmax(self.logits, axis=1))[1]




        #
        # Localisation Loss - WeightedSmoothL1LocalizationLoss
        #


        # top-1 loc,
        pred_class = tf.argmax(self.logits, axis=1)

        self.cam_t1 = self.get_classmap(pred_class, self.conv)
        self.pred_bbox, self.debug = utils.get_bbox_from_cam(self.cam_t1, 0.2)

        self.huber_loss_t1 = tf.losses.huber_loss(self.bbox,
                                                  self.pred_bbox)  # fixme: pred_cls 가 gt_cls 가 아닌 경우 loss 계산 다르게 해야함
        self.loc_loss_t1 = tf.reduce_mean(self.huber_loss_t1)

        self.iou_t1 = utils.calc_iou(self.pred_bbox, self.bbox)
        t1_iou_over_50 = self.iou_t1 > 0.5
        self.correct_and_iou_t1 = tf.logical_and(self.corrections, t1_iou_over_50)
        self.top1_loc_accuracy = tf.reduce_sum(tf.cast(self.correct_and_iou_t1, tf.float32)) / self.config.bs

        # gt-known loc, use this!
        self.cam_gt = self.get_classmap(self.y, self.conv)
        self.pred_bbox_gt, self.debug_gt = utils.get_bbox_from_cam(self.cam_gt, 0.2)

        self.huber_loss_gt = tf.losses.huber_loss(self.bbox, self.pred_bbox_gt)
        self.loc_loss_gt = tf.reduce_mean(self.huber_loss_gt)

        self.iou_gt = utils.calc_iou(self.pred_bbox_gt, self.bbox)

        gt_iou_over_50 = self.iou_gt > 0.5
        self.correct_and_iou_gt = gt_iou_over_50
        self.gt_known_loc_accuracy = tf.reduce_sum(tf.cast(self.correct_and_iou_gt, tf.float32)) / self.config.bs

        # Total Loss -- loc_loss_gt
        alpha = self.config.alpha
        beta = self.config.beta

        # self.tot_loss = alpha * self.cls_loss + beta * self.loc_loss_t1
        self.tot_loss = alpha * self.cls_loss + beta * self.loc_loss_gt

        # Optimizer
        if self.config.opt == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.config.beta1)
        elif self.config.opt == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        # grad-clipping
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_name)
        gvs = self.optimizer.compute_gradients(self.tot_loss, var_list=var_list)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs)

    def merge_summary(self):
        self.orig_image_sum = tf.summary.image("original_image", self.ox, max_outputs=4)

        self.image_sum = tf.summary.image("image", self.x, max_outputs=4)
        self.cam_gt_sum = tf.summary.image("cam_gt", tf.reshape(self.cam_gt, [-1, self.image_size, self.image_size, 1]),
                                           max_outputs=4)

        # self.cam_t1_sum = tf.summary.image("cam_t1", tf.reshape(self.cam_t1, [-1, self.image_size, self.image_size, 1]),
        #                                    max_outputs=4)


        # fixme: transpose..?         gt_bbox = tf.transpose(self.bbox, [1, 0, 3, 2]) / self.image_size

        _bbox = tf.cast(self.bbox / self.image_size, tf.float32)
        self.gt_bbox = tf.stack([_bbox[:, 1], _bbox[:, 0], _bbox[:, 3], _bbox[:, 2]], 1)

        _pbbox_gt = tf.cast(self.pred_bbox_gt / self.image_size, tf.float32)
        self.p_bbox_gt = tf.stack([_pbbox_gt[:, 1], _pbbox_gt[:, 0], _pbbox_gt[:, 3], _pbbox_gt[:, 2]], 1)

        _pbbox_t1 = tf.cast(self.pred_bbox / self.image_size, tf.float32)
        self.p_bbox_t1 = tf.stack([_pbbox_t1[:, 1], _pbbox_t1[:, 0], _pbbox_t1[:, 3], _pbbox_t1[:, 2]], 1)

        # self._bboxs = tf.stack([self.gt_bbox, self.p_bbox_gt, self.p_bbox_t1], axis=1)
        self._bboxs = tf.stack([self.gt_bbox, self.p_bbox_gt], axis=1)

        self.orig_image_bbox_sum = tf.summary.image("image_and_bbox",
                                                    tf.image.draw_bounding_boxes(self.ox, self._bboxs),
                                                    max_outputs=4)

        self.image_bbox_gt_sum = tf.summary.image("image_and_bbox_gt",
                                                  tf.image.draw_bounding_boxes(self.ox, tf.stack([self.gt_bbox], 1)),
                                                  max_outputs=4)

        self.cam_gt_bbox_sum = tf.summary.image("cam_gt_and_bbox",
                                                tf.image.draw_bounding_boxes(
                                                    tf.reshape(self.cam_gt, [-1, self.image_size, self.image_size, 1]),
                                                    self._bboxs),
                                                max_outputs=4)

        self.cls_loss_sum = tf.summary.scalar("cls_loss", self.cls_loss)
        self.loc_loss_t1_sum = tf.summary.scalar("loc_loss", self.loc_loss_t1)
        self.loc_loss_gt_sum = tf.summary.scalar("loc_loss_gt", self.loc_loss_gt)

        self.tot_loss_sum = tf.summary.scalar("tot_loss", self.tot_loss)

        self.cls_accuracy_sum = tf.summary.scalar("cls_accuracy", self.cls_accuracy)
        self.top1_loc_accuracy_sum = tf.summary.scalar("top1_loc_accuracy", self.top1_loc_accuracy)
        self.gt_knonwn_loc_accuracy_sum = tf.summary.scalar("gt_knonwn_loc_accuracy", self.gt_known_loc_accuracy)

        self.summary_merge = tf.summary.merge([
            self.image_sum,
            self.cam_gt_sum,
            self.orig_image_bbox_sum,
            self.image_bbox_gt_sum,
            self.cam_gt_bbox_sum,

            self.cls_loss_sum,
            self.loc_loss_t1_sum,
            self.loc_loss_gt_sum,
            self.tot_loss_sum,

            self.cls_accuracy_sum,
            self.top1_loc_accuracy_sum,
            self.gt_knonwn_loc_accuracy_sum]
        )
