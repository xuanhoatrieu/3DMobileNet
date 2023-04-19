import os
import csv
from sklearn.model_selection import train_test_split
import cv2

def read_all_video(root_path):
    folder_list = os.listdir(root_path)
    new_list = []
    for folder in folder_list:
        video_path = os.path.join(root_path, folder)
        num_frames = len(os.listdir(video_path))
        new_list.append([video_path, num_frames])
    return new_list

# data = read_all_video(root_path='Fall')
# data += read_all_video(root_path='No_Fall')
#
# train_data, test_data = train_test_split(data, test_size=0.1, shuffle=True)
#
# with open("train.csv", "w") as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerows(train_data)
#
# with open("test.csv", "w") as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerows(test_data)

folder_path = os.path.join("Fall", "Subject10Activity1Trial2Camera2")
file_list = sorted(os.listdir(folder_path))

for img_name in file_list:
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    cv2.imshow("img", img)
    cv2.waitKey(30)
