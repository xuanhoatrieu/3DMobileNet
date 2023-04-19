import os
import glob
import cv2
import numpy as np

root = 'No_Fall'

resize_value = 256
folder_list = sorted(os.listdir(root))

all_file = []
for foldername in folder_list:
    folderpath = os.path.join(root, foldername)
    filepaths = glob.glob(folderpath + '/*png')
    all_file += filepaths

for (i, filepath) in enumerate(all_file):
    if i%1000 == 0:
        print("---------------> Processing: ", i, "/", len(all_file))
    print('Current processing file: ', filepath)
    img = cv2.imread(filepath)
    height, width, _ = img.shape
    if width > height:
        if height > resize_value:
            scale = float(resize_value) / float(height)
            img = cv2.resize(img, (int(width * scale + 1), resize_value))
    else:
        if width > resize_value:
            scale = float(resize_value) / float(width)
            img = cv2.resize(img, (resize_value, int(height * scale + 1)))
    cv2.imwrite(filepath, img)








