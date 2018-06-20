import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

import utils

class ALEXNET():

    def __init__(self, config, inputs):

        self.config = config

        self.image_size = inputs.image_size
        self.class_num = inputs.class_num

        self.model_name = "AlexNet-has"

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
                                activation_fn=None,
                                weights_initializer=trunc_normal(0.005),
                                weights_regularizer=slim.l2_regularizer(0.001),
                                biases_initializer=tf.constant_initializer(0.1)
                                ):

                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.elu, updates_collections=None, is_training=self.is_training):

                    # modified alexnet
                    net = slim.conv2d(net, 64, [3, 3], scope='conv1')  # net = slim.conv2d(net, 64, [11, 11], 4, padding='VALID', scope='conv1')
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
                    net = slim.conv2d(net, 192, [3, 3], scope='conv2') # net = slim.conv2d(net, 192, [5, 5], scope='conv2')
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                    net = slim.conv2d(net, 384, [3, 3], scope='conv3')
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 384, [3, 3], scope='conv4')
                    net = slim.batch_norm(net, scope='bn4')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                    net = slim.batch_norm(net, scope='bn5')
                    net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
                    print(net.shape)

                    # alexnet-gap
                    net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                    net = slim.batch_norm(net, scope='bn6')
                    net = slim.conv2d(net, 1024, [3, 3], scope='conv7')

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

        # Classification Loss - SparseSoftmaxCrossEntropyLoss, WeightedSigmoidClassificationLoss

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.cross_entropy_loss = tf.reduce_mean(self.cross_entropy)
        self.cls_loss = self.cross_entropy_loss

        corrections = tf.equal(self.y, tf.argmax(self.logits, axis=1))
        self.cls_accuracy = tf.reduce_sum(tf.cast(corrections, tf.int32)) / tf.shape(self.y)[0]
        # self.accuracy = tf.metrics.accuracy(labels=self.y,
        #                                     predictions=tf.argmax(self.logits, axis=1))[1]


        #
        # Localisation Loss - WeightedSmoothL1LocalizationLoss
        #


        # top-1 loc,
        pred_class = tf.argmax(self.logits, axis=1)

        self.cam = self.get_classmap(pred_class, self.conv)

        # fixme:
        self.pred_bbox, self.debug = utils.get_bbox_from_cam(self.cam, 0.2)
        # self.pred_bbox = utils.get_bbox_from_cam(self.cam, 0.2)



        #############
        self.loc_loss = tf.reduce_mean(tf.losses.huber_loss(self.bbox, self.pred_bbox))

        self.iou = utils.calc_iou(self.pred_bbox, self.bbox)

        # fixme
        top_iou_over_50 = tf.greater(self.iou, 0.5)
        self.correct_and_iou = tf.logical_and(corrections, top_iou_over_50)
        self.top1_loc_accuracy = tf.reduce_sum(tf.cast(self.correct_and_iou, tf.float32)) / self.config.batch_size


        # gt-known loc
        self.cam_gt = self.get_classmap(self.y, self.conv)

        self.pred_bbox_gt, self.debug_gt = utils.get_bbox_from_cam(self.cam_gt, 0.2)

        self.loc_loss_gt = tf.reduce_mean(tf.losses.huber_loss(self.bbox, self.pred_bbox_gt))

        self.iou_gt = utils.calc_iou(self.pred_bbox_gt, self.bbox)

        gt_iou_over_50 = self.iou_gt > 0.5
        self.correct_and_iou_gt = tf.logical_and(corrections, gt_iou_over_50)
        self.gt_known_loc_accuracy = tf.reduce_sum(tf.cast(self.correct_and_iou_gt, tf.float32)) / self.config.batch_size


        # Total Loss -- loc_loss_gt
        alpha = 0
        self.tot_loss = self.cls_loss + alpha * self.loc_loss
        # self.tot_loss = self.cls_loss + alpha * self.loc_loss_gt


        # Optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.config.beta1)
        elif self.config.optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        # grad-clipping
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_name)
        gvs = self.optimizer.compute_gradients(self.tot_loss, var_list=var_list)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs)


    def merge_summary(self):
        self.image_sum = tf.summary.image("image", self.x, max_outputs=4)
        self.cam_sum = tf.summary.image("cam", tf.reshape(self.cam, [-1, self.image_size, self.image_size, 1]), max_outputs=4)
        self.cam_gt_sum = tf.summary.image("cam_gt", tf.reshape(self.cam_gt, [-1, self.image_size, self.image_size, 1]), max_outputs=4)
        # self.bbox_sum = tf.summary.image("bbox", self.bbox, max_outputs=4)
        # self.pbbox_sum = tf.summary.image("pbbox", self.pred_bbox, max_outputs=4)
        # self.pbbox_gt_sum = tf.summary.image("pbbox_gt", self.pred_bbox_gt, max_outputs=4)


        # fixme: transpose..?         gt_bbox = tf.transpose(self.bbox, [1, 0, 3, 2]) / self.image_size

        # fixme: tf.image.draw_bounding_boxes...?????
        # gt_bbox = self.bbox / self.image_size
        # p_bbox = self.pred_bbox / self.image_size

        self.gt_bbox = tf.stack([self.bbox[:, 1], self.bbox[:, 0],
                            self.bbox[:, 3], self.bbox[:, 2]], 1) / self.image_size
        self.p_bbox = tf.stack([self.pred_bbox[:, 1], self.pred_bbox[:, 0],
                           self.pred_bbox[:, 3], self.pred_bbox[:, 2]], 1) / self.image_size
        self.p_bbox_gt = tf.stack([self.pred_bbox_gt[:, 1], self.pred_bbox_gt[:, 0],
                              self.pred_bbox_gt[:, 3], self.pred_bbox_gt[:, 2]], 1) / self.image_size


        # gt_bbox = tf.stack([self.bbox[:, 1], self.bbox[:, 0],
        #                     self.bbox[:, 3], self.bbox[:, 2]], 1) / self.image_size
        # p_bbox = tf.stack([self.pred_bbox[:, 1], self.pred_bbox[:, 0],
        #                    self.pred_bbox[:, 3], self.pred_bbox[:, 2]], 1) / self.image_size
        # p_bbox_gt = tf.stack([self.pred_bbox_gt[:, 1], self.pred_bbox_gt[:, 0],
        #                       self.pred_bbox_gt[:, 3], self.pred_bbox_gt[:, 2]], 1) / self.image_size


        self._bboxs = tf.cast(tf.stack([self.gt_bbox], axis=1), tf.float32)
        self.image_bbox_sum = tf.summary.image("image_and_bbox", tf.image.draw_bounding_boxes(self.x, self._bboxs),
                                               max_outputs=4)

        # self._bboxs = tf.cast(tf.stack([self.gt_bbox, self.p_bbox], axis=1), tf.float32)
        # self.image_bbox_sum = tf.summary.image("image_and_bbox", tf.image.draw_bounding_boxes(self.x, self._bboxs),
        #                                        max_outputs=4)
        # bboxs_gt = tf.cast(tf.stack([gt_bbox, p_bbox_gt], axis=1), tf.float32)
        # self.image_bbox_gt_sum = tf.summary.image("image_and_bbox_gt", tf.image.draw_bounding_boxes(self.x, bboxs_gt),
        #                                           max_outputs=4)
        #
        # bboxs_all = tf.cast(tf.stack([self.gt_bbox, self.p_bbox, self.p_bbox_gt], axis=1), tf.float32)
        # self.image_bbox_all_sum = tf.summary.image("image_and_bbox_all", tf.image.draw_bounding_boxes(self.x, bboxs_all),
        #                                        max_outputs=4)

        self.cls_loss_sum = tf.summary.scalar("cls_loss", self.cls_loss)
        self.loc_loss_sum = tf.summary.scalar("loc_loss", self.loc_loss)
        self.loc_loss_gt_sum = tf.summary.scalar("loc_loss_gt", self.loc_loss_gt)

        self.tot_loss_sum = tf.summary.scalar("tot_loss", self.tot_loss)

        self.cls_accuracy_sum = tf.summary.scalar("cls_accuracy", self.cls_accuracy)
        self.top1_loc_accuracy_sum = tf.summary.scalar("top1_loc_accuracy", self.top1_loc_accuracy)
        self.gt_knonwn_loc_accuracy_sum = tf.summary.scalar("gt_knonwn_loc_accuracy", self.gt_known_loc_accuracy)

        self.summary_merge = tf.summary.merge([
                                               self.image_sum, self.cam_sum, self.cam_gt_sum,
                                               # self.bbox_sum, self.pbbox_sum, self.pbbox_gt_sum,
                                               self.image_bbox_sum,
            #                                    self.image_bbox_gt_sum,
            #                                    self.image_bbox_all_sum,
                                               self.cls_loss_sum,
                                               self.loc_loss_sum,
                                               self.loc_loss_gt_sum,
                                               self.tot_loss_sum,
                                               self.cls_accuracy_sum,
                                               self.top1_loc_accuracy_sum,
                                               self.gt_knonwn_loc_accuracy_sum]
                                              )

# original alexnet
# net = slim.conv2d(net, 64, [11, 11], 4, padding='VALID', scope='conv1')
# net = slim.batch_norm(net, scope='bn1')
# net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
# net = slim.conv2d(net, 192, [5, 5], scope='conv2')
# net = slim.batch_norm(net, scope='bn2')
# net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
# net = slim.conv2d(net, 384, [3, 3], scope='conv3')
# net = slim.batch_norm(net, scope='bn3')
# net = slim.conv2d(net, 384, [3, 3], scope='conv4')
# net = slim.batch_norm(net, scope='bn4')
# net = slim.conv2d(net, 256, [3, 3], scope='conv5')
# net = slim.batch_norm(net, scope='bn5')
# net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
