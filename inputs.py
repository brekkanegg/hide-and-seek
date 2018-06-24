import numpy as np
from glob import glob
from PIL import Image
import random
import pandas as pd
import re

# todo: augmentaion
# todo: hidden patch

class dataloader_tinyimagenet(object):
    def __init__(self, batch_size, mode='train', hide_prob=0.5):


        self.mode = mode
        self.image_size = 64
        self.class_num = 200
        self.hide_prob = 0.5

        if mode == 'train': #or mode == 'control':
            self.x = glob('data/tiny-imagenet-200/train/*/images/*.JPEG')
            # y = [re.search(r'train\/(.*?)\/images', ix).group(1) for ix in x]


        elif mode == 'val':
            self.x = glob('data/tiny-imagenet-200/val/images/*.JPEG')
            self.df = pd.read_csv('data/tiny-imagenet-200/val/val_annotations.txt', header=None, sep='\t', index_col=0)


        self.wnids = pd.read_csv('data/tiny-imagenet-200/wnids.txt', header=None)

        self.batch_size = batch_size
        self.data_count = len(self.x)
        self.num_batch = int(self.data_count / self.batch_size)
        self.pointer = 0


    def next_batch(self, patch_num=16):


        self.pointer = (self.pointer + 1) % self.num_batch

        start_pos = self.pointer * self.batch_size

        if self.mode == 'train':
            batch_images_dir = self.x[start_pos:start_pos + self.batch_size]
            temp_bi = [np.array(Image.open(_d))  # .resize([self.image_size, self.image_size])
                       if len(np.array(Image.open(_d)).shape) is 3
                       else to_rgb2(np.array(Image.open(_d)))
                       for _d in batch_images_dir]


            # hide patch
            batch_images_original = np.array(temp_bi)
            temp_bi = [hide_patch(i, patch_num, self.hide_prob) for i in temp_bi]

            batch_images = np.array(temp_bi)

            batch_labels = []
            batch_bboxs = []
            for xi in batch_images_dir:
                yi, eni = re.search(r'images\/(.*?)\.JPEG', xi).group(1).split('_')
                # fixme

                _yi = np.where(self.wnids[0] == yi)[0][0]
                batch_labels.append(_yi)
                bbox_txt = 'data/tiny-imagenet-200/train/{}/{}_boxes.txt'.format(yi, yi)
                with open(bbox_txt, 'r') as f:
                    line = f.readlines()[int(eni)]
                    bbox = re.search(r'JPEG(.*?)\n', line).group(1)
                    bbox = bbox.split('\t')[1:]
                    bbox = [int(b) for b in bbox]  # str -> int
                    batch_bboxs.append(bbox)

        if self.mode == 'val':
            batch_images_dir = self.x[start_pos:start_pos + self.batch_size]
            temp_bi = [np.array(Image.open(_d))  # .resize([self.image_size, self.image_size])
                       if len(np.array(Image.open(_d)).shape) is 3
                       else to_rgb2(np.array(Image.open(_d)))
                       for _d in batch_images_dir]

            batch_images = np.array(temp_bi)
            batch_images_original = batch_images

            batch_labels = []
            batch_bboxs = []
            for xi in batch_images_dir:
                eni = re.search(r'images\/(.*?)\.JPEG', xi).group(1)
                yi, bi = self.df.loc[eni + '.JPEG'][1], list(self.df.loc[eni + '.JPEG'][1:])
                _yi = np.where(self.wnids[0] == yi)[0][0]
                batch_labels.append(_yi)
                batch_bboxs.append(bi)

        return batch_images, batch_labels, batch_bboxs, batch_images_original

    def shuffle(self):
        random.seed(331)
        random.shuffle(self.x)


def hide_patch(img, patch_num=16, hide_prob=0.5, mean=127):
    # assume patch_num is int**2
    if patch_num == 1: return img

    pn = int(patch_num ** (1/2))
    patch_size = int(img.shape[1] // pn)
    patch_offsets = [(x * patch_size, y * patch_size) for x in range(pn) for y in range(pn)]

    for (px, py) in patch_offsets:
        if np.random.uniform() < hide_prob:
            # fixme: mean=127..?
            img[px:px + patch_size, py:py + patch_size] = mean

    return img


def to_rgb2(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = img[:, :, np.newaxis]
    return ret
