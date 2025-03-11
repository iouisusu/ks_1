from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import time
import codecs
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.param_attr import ParamAttr
from PIL import Image, ImageEnhance

paddle.enable_static()

target_size = [3, 512, 512]
mean_rgb = [127.5, 127.5, 127.5]
data_dir = "../data/data1"
eval_file = "eval.txt"
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
save_freeze_dir = "../code/freeze-model"
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_freeze_dir,
                                                                                      executor=exe)


# print(fetch_targets)


def crop_image(img, target_size):
    width, height = img.size
    p = min(target_size[2] / width, target_size[1] / height)
    resized_h = int(height * p)
    resized_w = int(width * p)
    img = img.resize((resized_w, resized_h), Image.BILINEAR)
    w_start = (resized_w - target_size[2]) / 2
    h_start = (resized_h - target_size[1]) / 2
    w_end = w_start + target_size[2]
    h_end = h_start + target_size[1]
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def resize_img(img, target_size):
    ret = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return ret


def read_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        # img = crop_image(img, target_size)
    img = resize_img(img, target_size)
    img = np.array(img).astype('float32')
    img -= mean_rgb
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    img = img[np.newaxis, :]
    return img


def infer(image_path):
    tensor_img = read_image(image_path)
    label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    return np.argmax(label)


def eval_all():
    eval_file_path = os.path.join(data_dir, eval_file)
    total_count = 0
    right_count = 0
    with codecs.open(eval_file_path, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        t1 = time.time()
        for line in lines:
            total_count += 1
            parts = line.strip().split()
            result = infer(parts[0])
            # print("infer result:{0} answer:{1}".format(result, parts[1]))
            if str(result) == parts[1]:
                right_count += 1
        period = time.time() - t1
        print("total eval count:{0} right eval count:{1} cost time:{2} predict accuracy:{3}".format(total_count,
                                                                                                    right_count,
                                                                                                    "%2.2f sec" % period,
                                                                                                    right_count / total_count))


if __name__ == '__main__':
    eval_all()