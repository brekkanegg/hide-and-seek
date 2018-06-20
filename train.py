import tensorflow as tf
import os, sys
import numpy as np
import time
import pprint

from alexnet import ALEXNET

import inputs

# Parameters
flags = tf.app.flags
flags.DEFINE_string("model", "alexnet", "alexnet of googlenet")

flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")

flags.DEFINE_float("learning_rate", 1e-2, "Learning rate of for optimizer")
flags.DEFINE_string("optimizer", "adam", "adam/rmsprop")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images [32]")

flags.DEFINE_integer("max_to_keep", 5, "model number of max to keep")

flags.DEFINE_bool("override", False, "Overriding checkpoint")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "save the image samples [samples]")
flags.DEFINE_string("summary_dir", "summary", "save the summary")

flags.DEFINE_integer("print_step", 100, "printing interval")
flags.DEFINE_integer("save_step", 1000, "saving_interval")

flags.DEFINE_string("gpu", "2", "# of gpu to use"),

FLAGS = flags.FLAGS

model_config = {'learning_rate': FLAGS.learning_rate,
                'optimizer': FLAGS.optimizer,
                'batch_size': FLAGS.batch_size,
                'model': FLAGS.model,
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

    train_inputs = inputs.dataloader_tinyimagenet(FLAGS.batch_size, mode='train')
    val_inputs = inputs.dataloader_tinyimagenet(FLAGS.batch_size, mode='val')

    print('Train Data Counts: ', train_inputs.data_count)

    # Model
    if FLAGS.model == 'alexnet':
        model = ALEXNET(config=FLAGS, inputs=train_inputs)

    sess.run(tf.local_variables_initializer())

    # Try Loading Checkpoint
    print('Checkpoint: ', FLAGS.checkpoint_dir)
    print(" [*] Reading Checkpoint...")
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path and not FLAGS.override:  # retrain
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        t1 = time.time()
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restoring Time: ', time.time() - t1)

        # graph = tf.get_default_graph()
        # todo: global step
        # tf.train.get_or_create_global_step(graph)

        print("""
======
An existing model was found in the checkpoint directory.
Loading...
======

        """)
    else:
        print("""
======
An existing model was not found in the checkpoint directory.
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


    batch_idxs = int(train_inputs.data_count // FLAGS.batch_size)

    patch_num_candidates = [4, 16, 64]
    for epoch in range(FLAGS.epoch):
        patch_num = np.random.choice(patch_num_candidates, p=[0.1, 0.3, 0.6])

        # fixme:
        if epoch == 15:
            FLAGS.learning_rate *= 1e-1
            print('Learning Rate Decreased to: ', FLAGS.learning_rate)
        if epoch == 35:
            FLAGS.learning_rate *= 1e-1
            print('Learning Rate Decreased to: ', FLAGS.learning_rate)

        train_inputs.shuffle()  # shuffle

        patch_num = 0
        for idx in range(0, batch_idxs):
            batch_xs, batch_ys, batch_bxs = train_inputs.next_batch(patch_num)

            feed = {model.x: batch_xs, model.y: batch_ys, model.bbox: batch_bxs,
                    model.is_training: True, model.learning_rate: FLAGS.learning_rate}

            ### debugging ###
            _x, _y, _logits = sess.run([model.x, model.y, model.logits], feed_dict=feed)

            # cam = sess.run(model.cam, feed_dict=feed)

            # bb, pbb, pbbgt, _bbx = sess.run([model.gt_bbox, model.p_bbox, model.p_bbox_gt, model._bboxs], feed_dict=feed)
            # print(bb, '\n', pbb, '\n', pbbgt, '\n', _bbx)

            # debug, debug_gt = sess.run([model.debug, model.debug_gt], feed_dict=feed)
            # cam, cam_gt = sess.run([model.cam, model.cam_gt], feed_dict=feed)
            # bbox, pbbox, pbbox_gt = sess.run([model.bbox, model.pred_bbox, model.pred_bbox_gt], feed_dict=feed)
            # print(bbox, '\n', pbbox, '\n', pbbox_gt)

            ### debugging ###


            cl, ll, tl, ca, tla, gla, s, _ = sess.run([model.cls_loss, model.loc_loss, model.tot_loss,
                                                       model.cls_accuracy,
                                                       model.top1_loc_accuracy, model.gt_known_loc_accuracy,
                                                       model.summary_merge, model.train_op],
                                                       feed_dict=feed)
            counter += 1

            if np.mod(counter, print_step) == 1:
                print("Epoch: [{:2d}] [{:4d}/{:4d}] [{:4d}] time: {:.4f}, "
                      "cls_accuracy:  {:.4f} cls_loss: {:.6f} loc_accuracy:  {:.4f} loc_loss: {:.6f}".format(
                      epoch, idx, batch_idxs, counter, time.time() - start_time, ca, cl, gla, ll))

        # todo: validation
            # Save model and sample image files / summary as well
            if np.mod(counter, save_step) == 1 and counter is not 1:
                summary_writer.add_summary(s, counter)
        #
        #         # validation
        #
        #         cls_loss = 0
        #         predictions = []
        #         labels = []
        #         val_batch_idxs = int(val_inputs.data_count // FLAGS.batch_size)
        #         for vi in range(0, val_batch_idxs):
        #
        #             # fixme
        #             try:
        #                 bv_xs, bv_ys = val_inputs.next_batch()
        #             except ValueError:
        #                 val_inputs.pointer += 1
        #                 bv_xs, bv_ys = val_inputs.next_batch()
        #
        #             cl, lg, lb = sess.run([model.cross_entropy_loss, model.logits, model.y],
        #                                   feed_dict={model.x: bv_xs, model.y: bv_ys, model.is_training: False})
        #
        #             cls_loss += cl
        #             predictions.extend(np.argmax(lg, 1))
        #             labels.extend(lb)
        #
        #         val_loss = cls_loss
        #         val_loss_sum = tf.Summary(value=[
        #             tf.Summary.Value(tag="val_loss", simple_value=val_loss),
        #         ])
        #         summary_writer.add_summary(val_loss_sum, counter)
        #
        #         val_acc = sum(np.equal(predictions, labels)) / len(labels)
        #         val_acc_sum = tf.Summary(value=[
        #             tf.Summary.Value(tag="val_acc", simple_value=val_acc),
        #         ])
        #         summary_writer.add_summary(val_acc_sum, counter)
        #         print("Validation Accuracy: {:.4f}".format(val_acc))
        #
        #         # if cls_loss < min_val_loss:
        #         if val_acc > max_val_acc:
        #             min_val_loss = cls_loss
        #             max_val_acc = val_acc
        #             saver.save(sess, FLAGS.checkpoint_dir + '/{}.ckpt'.format(model.model_name))
        #             print('Model saved at: {}/{}.ckpt'.format(FLAGS.checkpoint_dir, model.model_name))
        #             print('Max Val Accuracy: {}'.format(max_val_acc))
        #             pprint.pprint(model_config)
        #             # else:
        #             #     if a > max_val_acc + 0.5: ## overfitting
        #             #         sys.exit("Stop Training! Max Val Accuracy: {} Iteration: {} Time Spent: {:.4f}"
        #             #                  .format(max_val_acc, counter, time.time() - start_time))
        #
        #             # stop_stack += 1
        #             # print('Stop Stack: {}/100, Iterations: {}'.format(stop_stack, counter))
        #             # # model.learning_rate *= 0.1  # learning rate *10
        #             # if stop_stack == 100:
        #             #     sys.exit("Stop Training! Iteration: {} Time Spent: {}".format(counter, time.time() - start_time))
        #
        # print('Training finished')
        # pprint.pprint(model_config)