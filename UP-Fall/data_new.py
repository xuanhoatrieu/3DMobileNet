import os
import random
import numpy as np
import csv
from PIL import Image
import glob
import cv2
import tensorflow as tf

def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

def get_frames_for_sample(sample, num_frames_per_clip):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    folder_name = sample[0].decode('UTF-8')
    # images = sorted(glob.glob(os.path.join(folder_name, '/*jpg')))
    images = sorted(glob.glob(folder_name + '/*png'))
    start_idx = random.randint(0, len(images) - num_frames_per_clip - 1)
    if "No_Fall" in folder_name:
        label = 0.
    else:
        label = 1.
    return images, start_idx, label

def read_images(frames, start_idx, num_frames_per_clip):
    img_data = []
    for i in range(start_idx, start_idx + num_frames_per_clip):
        img = Image.open(frames[i])
        img = np.asarray(img)
        img_data.append(img)
    return img_data

def data_process(tmp_data, crop_size, is_train):
    img_datas = []
    crop_x = 0
    crop_y = 0

    if is_train and random.random()>0.5:
        flip = True
    else:
        flip = False

    if is_train and random.random()>0.8:
        cvt_color = True
    else:
        cvt_color = False

    if is_train and random.random()>0.5:
        channel1, channel2 = random.choices([0, 1, 2], k=2)
    else:
        channel1, channel2 = 0, 0

    size = crop_size
    # if is_train and crop_size==112:
    #     size = random.choice([129, 112, 96, 84])
        # size = random.choice([129, 112, 96])

    if is_train and crop_size==224:
        size = random.choice([256, 224, 192, 168])
        # size = random.choice([256, 224, 192])

    for j in range(len(tmp_data)):
        # img = Image.fromarray(tmp_data[j].astype(np.uint8))
        img = np.asarray(tmp_data[j], dtype=np.float32)
        # if img.width > img.height:
        #     scale = float(resize_value) / float(img.height)
        #     img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), resize_value))).astype(np.float32)
        # else:
        #     scale = float(resize_value) / float(img.width)
        #     img = np.array(cv2.resize(np.array(img), (resize_value, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            if is_train:
                crop_x = random.randint(0, int(img.shape[0] - size))
                crop_y = random.randint(0, int(img.shape[1] - size))
            else:
                crop_x = int((img.shape[0] - size) / 2)
                crop_y = int((img.shape[1] - size) / 2)
        img = img[crop_x:crop_x + size, crop_y:crop_y + size, :]
        img = np.array(cv2.resize(img, (crop_size, crop_size))).astype(np.float32)
        img = np.asarray(img) / 127.5
        img -= 1.

        if flip:
            img = np.flip(img, axis=1)

        if cvt_color:
            img = -img

        if channel1 != channel2:
            img = Channel_splitting(img, channel1, channel2)

        img_datas.append(img)
    return img_datas

def Channel_splitting(clip, channel1, channel2):
    clip[..., channel1] = clip[...,channel2]
    return clip

def generator_training_data(data, num_frames_per_clip=16, crop_size=224):
    while True:
        np.random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, start_idx, label = get_frames_for_sample(row, num_frames_per_clip)  # read all frames in video and length of the video
            clip = read_images(frames, start_idx, num_frames_per_clip)
            clip = data_process(clip, crop_size, is_train=True)
            clip = np.asarray(clip)
            yield clip, label

def data_augmentation(clip1, label):
    if random.random() > 0.5:
        alpha = random.uniform(0.5, 1.5)
        beta = random.uniform(-0.5, 0.5)
        clip1 = adjust_constrast_and_brightness(clip1, alpha, beta)
        clip1 = tf.clip_by_value(clip1, -1., 1.)

    if random.random() > 0.5:
        sigma = random.uniform(0.1, 2.0)
        clip1 = gaussian_blur(clip1, kernel_size=7, sigma=sigma)
        clip1 = tf.clip_by_value(clip1, -1., 1.)

    if random.random() > 0.5:
        hue_value = random.uniform(0, 0.1)
        clip1 = adjust_hue(clip1, hue_value)
        clip1 = tf.clip_by_value(clip1, -1., 1.)
    return clip1, label

def adjust_constrast_and_brightness(clip, alpha, beta):
    clip = clip * alpha + beta
    return clip

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    radius = tf.compat.v1.to_int32(kernel_size / 2)
    kernel_size = radius * 2 + 1
    x = tf.compat.v1.to_float(tf.range(-radius, radius + 1))
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.compat.v1.to_float(sigma), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred

def adjust_hue(clip, delta):
    clip = tf.image.adjust_hue(clip, delta=delta)
    return clip

def generator_test_data(data, num_frames_per_clip=16, crop_size=224):
    while True:
        np.random.shuffle(data)
        for i in range(len(data)):
            row = data[i]
            frames, start_idx, label = get_frames_for_sample(row, num_frames_per_clip)  # read all frames in video and length of the video
            clip = read_images(frames, start_idx, num_frames_per_clip)
            clip = data_process(clip, crop_size, is_train=False)
            clip = np.asarray(clip)
            yield clip, label