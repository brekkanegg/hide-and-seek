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

flags.DEFINE_float("lr", 1e-4, "Learning rate of for optimizer")
flags.DEFINE_float("alpha", 2, "Balancing hyperparameter of cls_loss")
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


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    test_inputs = inputs.dataloader_tinyimagenet(FLAGS.bs, mode='val', hide_prob=FLAGS.hp)

    print('Train Data Counts: ', test_inputs.data_count)

    # Model
    if FLAGS.model == 'alexnet':
        model = ALEXNET(config=FLAGS, inputs=test_inputs)
    elif FLAGS.model == 'alexnetmini':
        model = ALEXNETMini(config=FLAGS, inputs=test_inputs)

    elif FLAGS.model == 'googlenet':
        model = GOOGLENET(config=FLAGS, inputs=test_inputs)

    sess.run(tf.local_variables_initializer())

    # Try Loading Checkpoint
    print('Checkpoint: ', FLAGS.checkpoint_dir)
    print(" [*] Reading Checkpoint...")
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

     
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
    
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    # Train

    print('CONFIG: ')
    pprint.pprint(model_config)
    print('\nStart Testing')

    start_time = time.time()


    max_test_loc_gt_acc = 0
    max_test_loc_t1_acc = 0
    max_test_cls_acc = 0

    batch_idxs = int(test_inputs.data_count // FLAGS.bs)

    patch_num = 1


    test_inputs.shuffle()  # shuffle

    cls_loss = 0
    loc_gt_loss = 0
    loc_t1_loss = 0

    cls_corrections = []
    loc_gt_corrections = []
    loc_t1_corrections = []

    test_batch_idxs = int(test_inputs.data_count // FLAGS.bs)
    for vi in range(0, test_batch_idxs):

        bv_xs, bv_ys, bv_bxs, _ = test_inputs.next_batch()

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

    test_cls_acc = sum(cls_corrections) / len(cls_corrections)
    test_loc_gt_acc = sum(loc_gt_corrections) / len(loc_gt_corrections)
    test_loc_t1_acc = sum(loc_t1_corrections) / len(loc_t1_corrections)

    print('=== TEST ===')
    print("Time: {:4f}, cls_acc: {:.4f}, loc_gt_acc: {:.4f}, loc_t1_acc: {:.4f}, "
          .format(time.time() - start_time, test_cls_acc, test_loc_gt_acc, test_loc_t1_acc))

    # if cls_loss < min_test_loss:
    pprint.pprint(model_config)

    print('Test finished')
