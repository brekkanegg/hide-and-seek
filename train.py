import tensorflow as tf
import os, sys
import numpy as np
import time
import pprint

from alexnet import ALEXNET
from alexnetmini import ALEXNETMini
from googlenet import GOOGLENET

import inputs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore bounding box warning

# Parameters
flags = tf.app.flags
flags.DEFINE_string("model", "alexnet", "alexnet of googlenet")

flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_integer("cep", 10, "Epoch to train for classification")


flags.DEFINE_float("lr", 1e-4, "Learning rate of for optimizer")
flags.DEFINE_float("alpha", 1, "Balancing hyperparameter of cls_loss")
flags.DEFINE_float("beta", 1, "Balancing hyperparameter of loc_loss")

flags.DEFINE_bool("has", False, "hide patch")


flags.DEFINE_string("opt", "adam", "optimizer adam/rmsprop")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")

flags.DEFINE_float("cth", 0.3, "threshold for cam")
flags.DEFINE_float("hp", 0.5, "hide patch probability")


flags.DEFINE_integer("bs", 4, "The size of batch images [32]")

flags.DEFINE_integer("max_to_keep", 5, "model number of max to keep")

flags.DEFINE_bool("ov", False, "Overriding checkpoint")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "save the image samples [samples]")
flags.DEFINE_string("summary_dir", "summary", "save the summary")

flags.DEFINE_integer("print_step", 100, "printing interval")
flags.DEFINE_integer("save_step", 1000, "saving_interval")

flags.DEFINE_string("gpu", "1", "# of gpu to use"),

FLAGS = flags.FLAGS

model_config = {'learning_rate': FLAGS.lr,
                'alpha': FLAGS.alpha,
                'beta': FLAGS.beta,
                'optimizer': FLAGS.opt,
                'batch_size': FLAGS.bs,
                'model': FLAGS.model,
                'hide_patch': FLAGS.has,
                'cam_threshold': FLAGS.cth,
                'hide_prob': FLAGS.hp,
                }

model_dir = ['{}-{}'.format(key, model_config[key]) for key in sorted(model_config.keys())]
model_dir = '/'.join(model_dir)
print('CONFIG: ')
pprint.pprint(model_config)
FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
FLAGS.sample_dir = os.path.join(FLAGS.sample_dir, model_dir)
FLAGS.summary_dir = os.path.join(FLAGS.summary_dir, model_dir)

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.summary_dir):
    os.makedirs(FLAGS.summary_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    train_inputs = inputs.dataloader_tinyimagenet(FLAGS.bs, mode='train', hide_prob=FLAGS.hp)
    val_inputs = inputs.dataloader_tinyimagenet(FLAGS.bs, mode='val', hide_prob=FLAGS.hp)

    print('Train Data Counts: ', train_inputs.data_count)

    # Model
    if FLAGS.model == 'alexnet':
        model = ALEXNET(config=FLAGS, inputs=train_inputs)
    elif FLAGS.model == 'alexnetmini':
        model = ALEXNETMini(config=FLAGS, inputs=train_inputs)

    elif FLAGS.model == 'googlenet':
        model = GOOGLENET(config=FLAGS, inputs=train_inputs)

    sess.run(tf.local_variables_initializer())

    # Try Loading Checkpoint
    print('Checkpoint: ', FLAGS.checkpoint_dir)
    print(" [*] Reading Checkpoint...")
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path and not FLAGS.ov:  # retrain
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        t1 = time.time()
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restoring Time: ', time.time() - t1)

        # graph = tf.get_default_graph()
        # todo: global step
        # tf.train.get_or_create_global_step(graph)

        print("""
======
An existing model was 'found' in the checkpoint directory.
Loading...
======

        """)
    else:
        print("""
======
An existing model was 'not found' in the checkpoint directory.
Initializing a new one...
======
        """)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    # Train

    print('CONFIG: ')
    pprint.pprint(model_config)
    print('\nStart Training')

    start_time = time.time()


    # counter = tf.train.get_or_create_global_step(sess.graph)
    counter = 0
    epoch = 0

    save_step = FLAGS.save_step
    print_step = FLAGS.print_step

    max_val_loc_gt_acc = 0
    max_val_loc_t1_acc = 0
    max_val_cls_acc = 0

    batch_idxs = int(train_inputs.data_count // FLAGS.bs)

    for epoch in range(FLAGS.epoch):
        if FLAGS.has:
            patch_num_candidates = [16, 64]
            patch_num = np.random.choice(patch_num_candidates, p=[0.3, 0.7])
        else:
            patch_num = 1

        # fixme:
        if epoch == 40:
            FLAGS.lr *= 1e-1
            print('Learning Rate Decreased to: ', FLAGS.lr)
        # if epoch == 35:
        #     FLAGS.lr *= 1e-1
        #     print('Learning Rate Decreased to: ', FLAGS.learning_rate)




        ### debugging zone ###

        # batch_xs, batch_ys, batch_bxs, batch_oxs = train_inputs.next_batch(64)
        # #
        # feed = {model.x: batch_xs, model.y: batch_ys, model.bbox: batch_bxs, model.ox: batch_oxs,
        #         model.is_training: True, model.learning_rate: FLAGS.lr}
        #
        # bb, pbbgt, pbbt1= sess.run([model.gt_bbox, model.p_bbox_gt, model.p_bbox_t1], feed_dict=feed)
        # print(bb, '\n\n', pbbgt, '\n\n', pbbt1)



        #
        # dbs = sess.run(model.summary_merge, feed_dict=feed)
        # summary_writer.add_summary(dbs, counter)

        # _x, _y, _logits = sess.run([model.x, model.y, model.logits], feed_dict=feed)

        # cam = sess.run(model.cam, feed_dict=feed)

        #
        # print(bb, '\n', pbb, '\n', pbbgt, '\n', _bbx)

        # debug, debug_gt = sess.run([model.debug, model.debug_gt], feed_dict=feed)
        # cam, cam_gt = sess.run([model.cam, model.cam_gt], feed_dict=feed)
        # bbox, pbbox, pbbox_gt = sess.run([model.bbox, model.pred_bbox, model.pred_bbox_gt], feed_dict=feed)
        # print(bbox, '\n', pbbox, '\n', pbbox_gt)

        ######################

        train_inputs.shuffle()  # shuffle
        for idx in range(0, batch_idxs):
            batch_xs, batch_ys, batch_bxs, batch_oxs = train_inputs.next_batch(patch_num)

            feed = {model.x: batch_xs, model.y: batch_ys, model.bbox: batch_bxs, model.ox: batch_oxs,
                    model.is_training: True, model.learning_rate: FLAGS.lr}

            if epoch < FLAGS.cep:
                cl, t1l, gtl, tl, ca, t1la, gtla, s, _ = \
                    sess.run([model.cls_loss, model.loc_loss_t1, model.loc_loss_gt, model.tot_loss,
                              model.cls_accuracy, model.top1_loc_accuracy, model.gt_known_loc_accuracy,
                              model.summary_merge, model.cls_train_op],
                             feed_dict=feed)
            else:
                cl, t1l, gtl, tl, ca, t1la, gtla, s, _ = \
                    sess.run([model.cls_loss, model.loc_loss_t1, model.loc_loss_gt, model.tot_loss,
                              model.cls_accuracy, model.top1_loc_accuracy, model.gt_known_loc_accuracy,
                              model.summary_merge, model.train_op],
                             feed_dict=feed)

            counter += 1

            if np.mod(counter, print_step) == 1:
                print("Epoch: [{:2d}] [{:4d}/{:4d}] " 
                      "c_ls: {:.4f} gtl_ls: {:.4f} t1l_ls: {:.4f} c_ac: {:.4f} gtl_ac: {:.4f} t1l_ac: {:.4f}".format(
                      epoch, idx, batch_idxs, cl, gtl, t1l, ca, gtla, t1la))

            # Save model and sample image files / summary as well
            if np.mod(counter, save_step) == 1 and counter is not 1:
                summary_writer.add_summary(s, counter)


                # todo: loc accuracy

                # validation

                cls_loss = 0
                loc_gt_loss = 0
                loc_t1_loss = 0

                cls_corrections = []
                loc_gt_corrections = []
                loc_t1_corrections = []

                val_batch_idxs = int(val_inputs.data_count // FLAGS.bs)
                for vi in range(0, val_batch_idxs):

                    bv_xs, bv_ys, bv_bxs, _ = val_inputs.next_batch()

                    vfeed = {model.x: bv_xs, model.y: bv_ys, model.bbox: bv_bxs, model.is_training: False}

                    cl, vpred_class = sess.run([model.cls_loss, model.pred_class], feed_dict=vfeed)

                    vcls_l, vloc_gt_l, vloc_t1_l, vcls_cor, vloc_gt_cor, vloc_t1_cor = \
                        sess.run([model.cls_loss, model.loc_loss_gt, model.loc_loss_t1,
                                  model.corrections, model.correct_and_iou_gt, model.correct_and_iou_t1],
                                 feed_dict=vfeed)

                    cls_loss += vcls_l
                    loc_gt_loss += vloc_gt_l
                    loc_t1_loss += vloc_t1_l
                    cls_corrections.extend(vcls_cor)
                    loc_gt_corrections.extend(vloc_gt_cor)
                    loc_t1_corrections.extend(vloc_t1_cor)

                val_cls_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="val_cls_loss", simple_value=cls_loss)])
                summary_writer.add_summary(val_cls_loss_sum, counter)
                val_loc_gt_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loc_gt_loss", simple_value=loc_gt_loss)])
                summary_writer.add_summary(val_loc_gt_loss_sum, counter)
                val_loc_t1_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loc_t1_loss", simple_value=loc_t1_loss)])
                summary_writer.add_summary(val_loc_t1_loss_sum, counter)

                val_cls_acc = sum(cls_corrections) / len(cls_corrections)
                val_cls_acc_sum = tf.Summary(value=[tf.Summary.Value(tag="val_cls_acc", simple_value=val_cls_acc)])
                summary_writer.add_summary(val_cls_acc_sum, counter)
                val_loc_gt_acc = sum(loc_gt_corrections) / len(loc_gt_corrections)
                val_loc_gt_acc_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loc_gt_acc", simple_value=val_loc_gt_acc)])
                summary_writer.add_summary(val_loc_gt_loss_sum, counter)
                val_loc_t1_acc = sum(loc_t1_corrections) / len(loc_t1_corrections)
                val_loc_t1_acc_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loc_t1_acc", simple_value=val_loc_t1_acc)])
                summary_writer.add_summary(val_loc_t1_acc_sum, counter)

                print('=== VALIDATION ===')
                print("Time: {:4f}, cls_acc: {:.4f}, loc_gt_acc: {:.4f}, loc_t1_acc: {:.4f}, "
                      .format(time.time() - start_time, val_cls_acc, val_loc_gt_acc, val_loc_t1_acc))

                # if cls_loss < min_val_loss:


                if val_cls_acc > max_val_cls_acc:
                    max_val_cls_acc = val_cls_acc

                if val_loc_t1_acc > max_val_loc_t1_acc:
                    max_val_loc_t1_acc = val_loc_t1_acc

                if val_loc_gt_acc > max_val_loc_gt_acc:
                    max_val_loc_gt_acc = val_loc_gt_acc
                    saver.save(sess, FLAGS.checkpoint_dir + '/{}.ckpt'.format(model.model_name))
                    print('Model saved at: {}/{}.ckpt'.format(FLAGS.checkpoint_dir, model.model_name))

                print('Max Val Cls Acc: {:.4f} Max Val Loc GT Acc: {:.4f} Max Val Loc T1 Acc: {:.4f}'
                      .format(max_val_cls_acc, max_val_loc_gt_acc, max_val_loc_t1_acc))
                pprint.pprint(model_config)


    print('Training finished')
    pprint.pprint(model_config)