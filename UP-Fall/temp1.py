import os
import glob
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split

root = ['Fall', 'No_Fall']
full_dataset = []

for r in root:
    folder_list = os.listdir(r)
    for folder in folder_list:
        folder_path = os.path.join(r, folder)
        num_file = len(os.listdir(folder_path))
        full_dataset.append([folder_path, num_file])

train_set, test_set = train_test_split(full_dataset, test_size=0.2, random_state=2019)

with open("train.csv", "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(train_set)

with open("test.csv", "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(test_set)

