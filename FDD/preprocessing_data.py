import os
import glob
from subprocess import call

def get_all_folder(root_path):
    folders = []
    for all_file_folder in os.listdir(root_path):
        if os.path.isdir(all_file_folder) and (all_file_folder not in ".idea"):
            folders.append(root_path + all_file_folder + "/Videos/")
    return sorted(folders)

def extract_img(folder_path):
    files = os.listdir(folder_path)
    video_files = []
    file_no_ext = []
    for f in files:
        video_files.append(folder_path + f)
        file_no_ext.append(folder_path + f.split(".")[0])

    for i in range(len(video_files)):
        if not os.path.exists(file_no_ext[i]):
            os.mkdir(file_no_ext[i])
        src = video_files[i]
        dst = file_no_ext[i] + '/%05d.jpg'
        call(["ffmpeg", "-i", src, dst])


folders = get_all_folder("./")
# for folder in folders:
#     extract_img(folder)
extract_img("Coffee_room/Videos/")