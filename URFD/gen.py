import numpy as np
import os
import random
from PIL import Image
import pandas as pd
import cv2
import csv

def check_label(start_frame, end_frame, start_falling, end_falling):
    mid_frame = (start_frame + end_frame) // 2
    if start_falling < mid_frame and mid_frame < end_falling:
        return 1
    else:
        return 0

def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

def read_csvfile(filepath, num_frames_per_clip):
    new_data = []
    df = pd.read_csv(filepath)
    folder_path = df['folder_path'].to_list()
    num_frame = df['num_frames'].to_list()
    falling_frame = df['falling_frame'].to_list()
    num_video = len(folder_path)
    for video in range(num_video):
        frames = num_frame[video]
        if frames % num_frames_per_clip ==0:
            frames-=1
        num_clips = frames // num_frames_per_clip
        path = folder_path[video] + '-cam0-rgb'
        start_falling = falling_frame[video]
        end_falling = frames
        for i in range(num_clips):
            clip = {}
            start_frame = i * num_frames_per_clip
            end_frame = (i+1) * num_frames_per_clip
            if path[:3] =='adl':
                label=0
            else:
                label=check_label(start_frame,end_frame,start_falling, end_falling)
            clip['start_frame'] = start_frame
            clip['end_frame'] = end_frame
            clip['video_path'] = path
            clip['label'] = label
            new_data.append(clip)
        if path[:4]== 'fall':
            for i in range(start_falling, end_falling - num_frames_per_clip):
                clip = {}
                start_frame = i
                end_frame = start_frame + num_frames_per_clip
                label = 1
                clip['start_frame'] = start_frame
                clip['end_frame'] = end_frame
                clip['video_path'] = path
                clip['label'] = label
                new_data.append(clip)
    return new_data

# video_path,start_frame,end_frame,label

def count(data):
    no_falling = 0
    falling = 0
    for d in data:
        print(d)
        if int(d[3])==0:
            no_falling+=1
        else:
            falling+=1
    return falling, no_falling

def read_imgs(start_frame, end_frame, rgb_folder):
    rgb_clip = []
    for i in range(start_frame, end_frame):
        rgb_file = os.path.join('dataset', rgb_folder, rgb_folder + '-' + str(i + 1).zfill(3) + ".png")
        rgb_img = Image.open(rgb_file)
        rgb_img = rgb_img.resize((224,224))
        rgb_img = np.asarray(rgb_img) / 255.0
        rgb_clip.append(rgb_img)

    return rgb_clip

def generator_data(data, batch_size):
    while True:
        random.shuffle(data)
        X_rgb = []
        y = []
        for _ in range(batch_size):
            row = random.choice(data)
            start_frame = int(row[1])
            end_frame = int(row[2])
            folder= row[0]
            rgb_clip = read_imgs(start_frame, end_frame, folder)
            X_rgb.append(rgb_clip)
            y.append(int(row[3]))
        X_rgb = np.asarray(X_rgb)
        y = np.asarray(y)
        yield X_rgb, y

def test_data(data, batch_size):
    while True:
        random.shuffle(data)
        X_rgb = []
        y = []
        for i in range(0, len(data), batch_size):
            row = data[i]
            start_frame = int(row[1])
            end_frame = int(row[2])
            rgb_folder= row[0]
            rgb_clip= read_imgs(start_frame, end_frame, rgb_folder)
            X_rgb.append(rgb_clip)
            y.append(int(row[3]))
        X_rgb = np.asarray(X_rgb)
        y = np.asarray(y)
        yield X_rgb, y