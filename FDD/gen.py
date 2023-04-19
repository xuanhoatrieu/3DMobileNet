import numpy as np
import os
import random
from PIL import Image

def read_txt_files(input_folder):
    data = []
    annotation_folder = os.path.join(input_folder, "Annotation_files")
    video_folder = os.path.join(input_folder, "Videos")
    annotation_files = os.listdir(annotation_folder)
    for file in annotation_files:
        if 'txt' in file:
            object_data = {}
            object_data['annotation_path'] = os.path.join(annotation_folder, file)
            f = open(os.path.join(annotation_folder, file))
            start_falling = f.readline()
            end_falling = f.readline()
            object_data['start_falling']= int(start_falling)
            object_data['end_falling'] = int(end_falling)
            f.close()
            # object_data['rgb_folder'] = os.path.join(video_folder, file[:-4] + "_RGB")
            # object_data['mhi_folder'] = os.path.join(video_folder, file[:-4] + "_MHI")
            # object_data['num_frames'] = len(os.listdir(os.path.join(video_folder, file[:-4] + "_RGB")))
            object_data['rgb_folder'] = os.path.join(video_folder, file[:-4])
            object_data['mhi_folder'] = os.path.join(video_folder, file[:-4] + "_MHI")
            object_data['num_frames'] = len(os.listdir(os.path.join(video_folder, file[:-4])))
            data.append(object_data)
    return data

def check_label(start_frame, end_frame, start_falling, end_falling):
    mid_frame = (start_frame + end_frame) // 2
    if start_falling < mid_frame and mid_frame < end_falling:
        return 1
    else:
        return 0

def video_split(data, num_frames_per_clip):
    new_data = []
    for d in data:
        start_falling = d['start_falling']
        end_falling = d['end_falling']
        num_frames = d['num_frames']
        rgb_folder = d['rgb_folder']
        mhi_folder = d['mhi_folder']
        if num_frames % num_frames_per_clip == 0:
            num_frames-=1
        num_clips = num_frames // num_frames_per_clip
        for i in range(num_clips):
            clip = {}
            start_frame = i * num_frames_per_clip
            end_frame = (i+1) * num_frames_per_clip
            label = check_label(start_frame, end_frame, start_falling, end_falling)
            clip['rgb_folder'] = rgb_folder
            clip['mhi_folder'] = mhi_folder
            clip['start_frame'] = start_frame
            clip['end_frame'] = end_frame
            clip['label'] = label
            new_data.append(clip)
        for i in range(start_falling,end_falling-num_frames_per_clip):
            clip = {}
            start_frame = i
            end_frame = start_frame + num_frames_per_clip
            label = 1
            clip['rgb_folder'] = rgb_folder
            clip['mhi_folder'] = mhi_folder
            clip['start_frame'] = start_frame
            clip['end_frame'] = end_frame
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

def read_imgs(start_frame, end_frame, rgb_folder):
    rgb_clip = []
    for i in range(start_frame, end_frame):
        rgb_file = os.path.join(rgb_folder, str(i + 1).zfill(5) + ".jpg")
        rgb_img = Image.open(rgb_file)
        rgb_img = rgb_img.resize((224,224))
        rgb_img = np.asarray(rgb_img) / 255.0
        rgb_clip.append(rgb_img)

    return rgb_clip

def generator_data(data, batch_size):
    while True:
        X_rgb = []
        y = []
        for _ in range(batch_size):
            row = random.choice(data)
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            rgb_folder= row['rgb_folder']
            rgb_clip = read_imgs(start_frame, end_frame, rgb_folder)
            X_rgb.append(rgb_clip)
            y.append(row['label'])
        X_rgb = np.asarray(X_rgb)
        y = np.asarray(y)
        yield X_rgb, y

def test_data(data, batch_size=32):
    while True:
        X_rgb = []
        y = []
        for i in range(0, len(data), batch_size):
            row = data[i]
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            rgb_folder = row['rgb_folder']
            rgb_clip= read_imgs(start_frame, end_frame, rgb_folder)
            X_rgb.append(rgb_clip)
            y.append(row['label'])
        X_rgb = np.asarray(X_rgb)
        y = np.asarray(y)
        yield X_rgb, y