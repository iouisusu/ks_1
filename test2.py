# -*- coding: UTF-8 -*-
"""
训练常用视觉基础网络，用于分类任务
需要将训练图片，类别文件 label_list.txt 放置在同一个文件夹下
程序会先读取 train.txt 文件获取类别数和图片数量
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.75'
import numpy as np
import time
import math
import paddle
import paddle.fluid as fluid
import codecs
import logging

from paddle.fluid.initializer import MSRA
from paddle.fluid.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr
from PIL import Image
from PIL import ImageEnhance

paddle.enable_static()

train_parameters = {
    "input_size": [3, 512, 512],
    "class_dim": -1,  # 分类数，会在初始化自定义 reader 的时候获得
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得
    "label_dict": {},
    "data_dir": "../data/data1",  # 训练数据存储地址
    "train_file_list": "train.txt",
    "label_file": "label_list.txt",
    "save_freeze_dir": "./freeze-model",
    "save_persistable_dir": "./persistable-params",
    "continue_train": True,  # 是否接着上一次保存的参数接着训练，优先级高于预训练模型
    "pretrained": False,  # 是否使用预训练的模型
    "pretrained_dir": "data/data6593/DenseNet_pretrained",
    "mode": "train",
    "num_epochs": 5,  # ----------120,
    "train_batch_size": 4,  # ----------12,
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值
    "use_gpu": True,
    "dropout_prob": 0.2,
    "dropout_seed": None,
    "image_enhance_strategy": {  # 图像增强相关策略
        "need_distort": False,  # True,  # 是否启用图像颜色增强
        "need_rotate": False,  # True,  # 是否需要增加随机角度
        "need_crop": True,  # 是否要增加裁剪
        "need_flip": False,  # True,  # 是否要增加水平随机翻转
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125,
        "rotate_prob": 0.5,
        "rotate_range": 14
    },
    # "image_enhance_strategy": {  # 简化图像增强相关策略
    #     "need_distort": False,
    #     "need_rotate": False,
    #     "need_crop": False,
    #     "need_flip": False,
    #     "hue_prob": 0,
    #     "contrast_prob": 0,
    #     "saturation_prob": 0,
    #     "brightness_prob": 0,
    #     "rotate_prob": 0
    # },
    "early_stop": {
        "sample_frequency": 20,  # 减少到每20个batch检查一次50,
        "successive_limit": 3,  # 减少到3次 5,
        "good_acc1": 0.60  # 降低到60%0.90
    },
    # "rsm_strategy": {
    #     "learning_rate": 0.002,
    #     "lr_epochs": [20, 40, 60, 80, 100],
    #     "lr_decay": [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    # }
    "rsm_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [5],  # 在第5个epoch调整一次
        "lr_decay": [1, 0.5]
    }
}


class DenseNet():
    def __init__(self, layers, dropout_prob):
        self.layers = layers
        self.dropout_prob = dropout_prob

    def bottleneck_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num * 4, filter_size=1, name=name + '_conv1')
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)

        bn = fluid.layers.batch_norm(input=dropout, act='relu', name=name + '_bn2')
        conv2 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=3, padding=1, name=name + '_conv2')
        dropout = fluid.layers.dropout(x=conv2, dropout_prob=self.dropout_prob)

        return dropout

    def dense_block(self, input, block_num, fliter_num, name):
        layers = []
        layers.append(input)  # 拼接到列表

        x = self.bottleneck_layer(input, fliter_num, name=name + '_bottle_' + str(0))
        layers.append(x)
        for i in range(block_num - 1):
            x = paddle.fluid.layers.concat(layers, axis=1)
            x = self.bottleneck_layer(x, fliter_num, name=name + '_bottle_' + str(i + 1))
            layers.append(x)

        return paddle.fluid.layers.concat(layers, axis=1)

    def transition_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=1, name=name + '_conv1')
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)

        return fluid.layers.pool2d(input=dropout, pool_size=2, pool_type='avg', pool_stride=2)

    def net(self, input, class_dim=1000):

        layer_count_dict = {
            121: (32, [6, 12, 24, 16]),
            169: (32, [6, 12, 32, 32]),
            201: (32, [6, 12, 48, 32]),
            161: (48, [6, 12, 36, 24])
        }
        layer_conf = layer_count_dict[self.layers]

        conv = fluid.layers.conv2d(input=input, num_filters=layer_conf[0] * 2,
                                   filter_size=7, stride=2, padding=3, name='densenet_conv0')
        conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_padding=1, pool_type='max', pool_stride=2)
        for i in range(len(layer_conf[1]) - 1):
            conv = self.dense_block(conv, layer_conf[1][i], layer_conf[0], 'dense_' + str(i))
            conv = self.transition_layer(conv, layer_conf[0], name='trans_' + str(i))

        conv = self.dense_block(conv, layer_conf[1][-1], layer_conf[0], 'dense_' + str(len(layer_conf[1])))
        conv = fluid.layers.pool2d(input=conv, global_pooling=True, pool_type='avg')
        out = fluid.layers.fc(conv, class_dim, act='softmax')
        # last fc layer is "out"
        return out


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train1.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    train_file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_file_list'])
    label_list = os.path.join(train_parameters['data_dir'], train_parameters['label_file'])
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.strip().split()
            train_parameters['label_dict'][parts[1]] = int(parts[0])
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(train_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)


def resize_img(img, target_size):
    """
    强制缩放图片
    :param img:
    :param target_size:
    :return:
    """
    # target_size = input_size
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


def random_crop(img, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w ** 2),
                (float(img.size[1]) / img.size[0]) / (h ** 2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((train_parameters['input_size'][1], train_parameters['input_size'][2]), Image.BILINEAR)
    return img


def rotate_image(img):
    """
    图像增强，增加随机旋转角度
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['rotate_prob']:
        range = train_parameters['image_enhance_strategy']['rotate_range']
        angle = np.random.randint(-range, range)
        img = img.rotate(angle)
    return img


def random_brightness(img):
    """
    图像增强，亮度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    """
    图像增强，对比度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    """
    图像增强，饱和度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    """
    图像增强，色度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['hue_prob']:
        hue_delta = train_parameters['image_enhance_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_color(img):
    """
    概率的图像增强
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.35:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    elif prob < 0.7:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def custom_image_reader(file_list, data_dir, mode):
    """
    自定义用户图片读取器，先初始化图片种类，数量
    :param file_list:
    :param data_dir:
    :param mode:
    :return:
    """
    with codecs.open(file_list) as flist:
        lines = [line.strip() for line in flist]

    def reader():
        np.random.shuffle(lines)
        for line in lines:
            if mode == 'train' or mode == 'val':
                img_path, label = line.split()
                img = Image.open(img_path)
                try:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if train_parameters['image_enhance_strategy']['need_distort'] == True:
                        img = distort_color(img)
                    if train_parameters['image_enhance_strategy']['need_rotate'] == True:
                        img = rotate_image(img)
                    if train_parameters['image_enhance_strategy']['need_crop'] == True:
                        img = random_crop(img, train_parameters['input_size'])
                    if train_parameters['image_enhance_strategy']['need_flip'] == True:
                        mirror = int(np.random.uniform(0, 2))
                        if mirror == 1:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            # HWC--->CHW && normalized
                    img = np.array(img).astype('float32')
                    img -= train_parameters['mean_rgb']
                    img = img.transpose((2, 0, 1))  # HWC to CHW
                    img *= 0.007843  # 像素值归一化
                    yield img, int(label)
                except Exception as e:
                    pass  # 以防某些图片读取处理出错，加异常处理
            elif mode == 'test':
                img_path = os.path.join(data_dir, line)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = resize_img(img, train_parameters['input_size'])
                # HWC--->CHW && normalized
                img = np.array(img).astype('float32')
                img -= train_parameters['mean_rgb']
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img *= 0.007843  # 像素值归一化
                yield img

    return reader


def optimizer_rms_setting():
    """
    阶梯型的学习率适合比较大规模的训练数据
    """
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    learning_strategy = train_parameters['rsm_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values))

    return optimizer


def load_params(exe, program):
    if train_parameters['continue_train'] and os.path.exists(train_parameters['save_persistable_dir']):
        logger.info('load params from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_persistable_dir'],
                                   main_program=program)
    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_dir']):
        logger.info('load params from pretrained model')

        def if_exist(var):
            return os.path.exists(os.path.join(train_parameters['pretrained_dir'], var.name))

        fluid.io.load_vars(exe, train_parameters['pretrained_dir'], main_program=program,
                           predicate=if_exist)


def train():
    train_prog = fluid.Program()
    train_startup = fluid.Program()
    logger.info("create prog success")
    logger.info("train config: %s", str(train_parameters))
    logger.info("build input custom reader and data feeder")
    file_list = os.path.join(train_parameters['data_dir'], "train.txt")
    mode = train_parameters['mode']
    batch_reader = paddle.batch(custom_image_reader(file_list, train_parameters['data_dir'], mode),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=False)
    batch_reader = paddle.reader.shuffle(batch_reader, train_parameters['train_batch_size'])
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
    # 定义输入数据的占位符
    img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    # 选取不同的网络
    logger.info("build newwork")
    model = DenseNet(121, train_parameters['dropout_prob'])
    out = model.net(input=img, class_dim=train_parameters['class_dim'])

    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    optimizer = optimizer_rms_setting()
    optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    main_program = fluid.default_main_program()
    exe.run(fluid.default_startup_program())
    train_fetch_list = [avg_cost.name, acc_top1.name, out.name]

    load_params(exe, main_program)

    # 训练循环主体
    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    good_acc1 = stop_strategy['good_acc1']
    successive_count = 0
    stop_train = False
    total_batch_count = 0
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: %d, start read image", pass_id)
        batch_id = 0
        for step_id, data in enumerate(batch_reader()):
            t1 = time.time()
            loss, acc1, pred_ot = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=train_fetch_list)
            t2 = time.time()
            batch_id += 1
            total_batch_count += 1
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            if batch_id % 10 == 0:
                logger.info(
                    "Pass {0}, trainbatch {1}, loss {2}, acc1 {3}, time {4}".format(pass_id, batch_id, loss, acc1,
                                                                                    "%2.2f sec" % period))
                # 简单的提前停止策略，认为连续达到某个准确率就可以停止了
            if acc1 >= good_acc1:
                successive_count += 1
                logger.info(
                    "current acc1 {0} meets good {1}, successive count {2}".format(acc1, good_acc1, successive_count))
                fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                              feeded_var_names=['img'],
                                              target_vars=[out],
                                              main_program=main_program.clone(for_test=True),
                                              executor=exe)
                if successive_count >= successive_limit:
                    logger.info("end training")
                    stop_train = True
                    break
            else:
                successive_count = 0

                # 通用的保存策略，减小意外停止的损失
            if total_batch_count % sample_freq == 0:
                logger.info("temp save {0} batch train result, current acc1 {1}".format(total_batch_count, acc1))
                fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                                           main_program=main_program,
                                           executor=exe)
        if stop_train:
            break
    logger.info("training till last epcho, end training")
    fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                               main_program=main_program,
                               executor=exe)
    fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                  feeded_var_names=['img'],
                                  target_vars=[out],
                                  main_program=main_program.clone(for_test=True),
                                  executor=exe)


if __name__ == '__main__':
    init_log_config()
    init_train_parameters()
    train()
