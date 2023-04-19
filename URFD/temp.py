import glob
import os

folder_name = 'fall-03-cam0-rgb'

images = sorted(glob.glob(os.path.join('dataset', folder_name + '/*png')))

print(images)