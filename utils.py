import tensorflow as tf

def calc_iou(pred_bbox, gt_bbox):

    min_px, min_py, max_px, max_py = pred_bbox[:, 0], pred_bbox[:, 1], pred_bbox[:, 2], pred_bbox[:, 3]
    min_gx, min_gy, max_gx, max_gy = gt_bbox[:, 0], gt_bbox[:, 1], gt_bbox[:, 2], gt_bbox[:, 3]

    a1 = (max_px - min_px) * (max_py - min_py)
    a2 = (max_gx - min_gx) * (max_gy - min_gy)
    inter = (tf.minimum(max_gx, max_px) - tf.maximum(min_gx, min_px)) * \
            (tf.minimum(max_gy, max_py) - tf.maximum(min_gy, min_py))
    union = a1 + a2 - inter
    iou = inter / union

    return iou


def get_bbox_from_cam(cam, threshold=0.2):

    th = tf.reduce_max(cam) * threshold

    bm = tf.greater(cam, th)  # expecting binary mask

    bm_x = tf.reduce_any(bm, axis=1)
    bm_y = tf.reduce_any(bm, axis=2)

    # fixme: batch 로 받기..?
    min_px = tf.map_fn(lambda x: tf.reduce_min(tf.where(x)[:, 0]), bm_x, dtype=tf.int64)
    min_py = tf.map_fn(lambda x: tf.reduce_min(tf.where(x)[:, 0]), bm_y, dtype=tf.int64)
    max_px = tf.map_fn(lambda x: tf.reduce_max(tf.where(x)[:, 0]), bm_x, dtype=tf.int64)
    max_py = tf.map_fn(lambda x: tf.reduce_max(tf.where(x)[:, 0]), bm_y, dtype=tf.int64)

    # fixme: 없는 경우 대비해서 일단 이렇게
    min_px = tf.clip_by_value(min_px, 0, 63)
    min_py = tf.clip_by_value(min_py, 0, 63)
    max_px = tf.clip_by_value(max_px, 0, 63)
    max_py = tf.clip_by_value(max_py, 0, 63)

    _bbox = [min_px, min_py, max_px, max_py]

    bbox = tf.stack(_bbox, 1)

    return bbox, (min_px, min_py, bm, bm_x, th, cam)



