import numpy as np
import os
import random
from PIL import Image
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

def check_label(start_frame, end_frame, start_falling, end_falling):
    mid_frame = (start_frame + end_frame) // 2
    if start_falling < mid_frame and mid_frame < end_falling:
        return 1
    else:
        return 0

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

def count(data):
    no_falling = 0
    falling = 0
    for d in data:
        if d['label']==0:
            no_falling+=1
        else:
            falling+=1
    return falling, no_falling

new_data = read_csvfile('full.csv', 16)
new_data = random.shuffle(new_data)

data_train, data_test = train_test_split(new_data, test_size=0.2, random_state=2019)
print(count(data_train), len(data_train))
print(count(data_test), len(data_test))

print(new_data)

file_train = open('train_3dmobile.csv', 'w')

writer = csv.writer(file_train)

writer.writerow(data_train)
file_train.close()

file_test = open('test_3dmobile.csv', 'w')

writer = csv.writer(file_test)

writer.writerow(file_test)
file_test.close()
